"""Manual three-stage Joint+history hyperparameter study; no architecture grid."""

import argparse
import fcntl
import hashlib
import json
import platform
import re
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from spend_baselines.suite_data import BASE_COLUMNS, atomic_json, digest, row_weights, split_indices
from spend_baselines.suite_models import Neural
from spend_behavior.models import inner_date

from .data import FrozenRun, TuningData


class BudgetReached(RuntimeError):
    pass


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg["version"] != 1 or cfg["seeds"] != [42, 123, 2026] or cfg["screen_seed"] != 42:
        raise ValueError("Use the fixed mean3 seeds and seed42 screening")
    early, late = pd.to_datetime(cfg["tuning_dates"]), pd.to_datetime(cfg["confirmation_dates"])
    if not len(early) or not len(late) or early.has_duplicates or late.has_duplicates:
        raise ValueError("Empty/duplicate dates")
    if early.max() + pd.Timedelta(days=30) > late.min():
        raise ValueError("Tuning labels must be available before confirmation")
    if cfg["base_settings"]["hidden_sizes"] != [128, 64]:
        raise ValueError("The best current architecture is fixed at 128/64")
    names = [p["name"] for p in cfg["profiles"]]
    if len(names) != len(set(names)) or not 2 <= len(names) <= 8 or names[0] != "baseline":
        raise ValueError("Use baseline and at most seven explicit new profiles")
    if cfg["profiles"][0]["overrides"] or not 1 <= cfg["shortlist_size"] <= min(2, len(names)-1):
        raise ValueError("Baseline is frozen; shortlist is at most two profiles")
    allowed = {"learning_rate", "weight_decay", "dropout", "batch_size"}
    for p in cfg["profiles"]:
        if not re.fullmatch(r"[a-z0-9_]+", p["name"]) or not set(p["overrides"]) <= allowed:
            raise ValueError("Invalid profile name/parameter")
        s = {**cfg["base_settings"], **p["overrides"]}
        if s["learning_rate"] <= 0 or s["weight_decay"] < 0 or not 0 <= s["dropout"] < 1 or s["batch_size"] < 1:
            raise ValueError("Invalid optimiser settings")
    if cfg["minimum_tuning_gain"] < 0 or cfg["bootstrap_repeats"] < 1:
        raise ValueError("Invalid gain/bootstrap")
    return cfg


def budget(cfg):
    screen = (len(cfg["profiles"])-1) * len(cfg["tuning_dates"])
    shortlist = cfg["shortlist_size"] * (len(cfg["seeds"])-1) * len(cfg["tuning_dates"])
    confirm = len(cfg["seeds"]) * len(cfg["confirmation_dates"])
    return {"screen_seed_evaluations": screen, "shortlist_additional_seed_evaluations": shortlist,
            "confirmation_max_seed_evaluations": confirm,
            "max_seed_evaluations": screen+shortlist+confirm, "max_neural_fits": 2*(screen+shortlist+confirm),
            "control_retrained": False, "embeddings": False}


def temporal_split(data, cutoff):
    objects = data.objects
    outer = split_indices(objects, "D2", cutoff)
    outer = outer[data.active[outer]]
    date = inner_date(objects, cutoff)
    inner = split_indices(objects, "D2", date)
    inner = inner[data.active[inner]]
    iv = data.valid(date)
    if not len(outer) or not len(inner) or not len(iv):
        raise ValueError("Empty purged temporal split")
    assert objects.iloc[outer].target_end.max() <= pd.Timestamp(cutoff)
    assert objects.iloc[inner].target_end.max() <= date
    return outer, inner, iv, date


def pooled(frame, dates):
    a = frame[frame.date.isin(dates)]
    if set(a.date) != set(dates) or a.Id.duplicated().any():
        raise ValueError("Incomplete/duplicate tuning predictions")
    if not np.isfinite(a[["z", "target_z"]]).all().all():
        raise ValueError("Non-finite tuning predictions")
    return float(np.sqrt(np.mean((a.z-a.target_z)**2)))


def choose_shortlist(frames, cfg):
    names = [p["name"] for p in cfg["profiles"] if p["name"] != "baseline"]
    if set(frames) != set(names):
        raise ValueError("All seven/fixed profiles must finish seed42 screening")
    ranked = sorted(({"profile": n, "RMSLE": pooled(frames[n], cfg["tuning_dates"])} for n in names),
                    key=lambda r: (r["RMSLE"], r["profile"]))
    return {"tuning_dates": cfg["tuning_dates"], "screen_seed": cfg["screen_seed"],
            "ranking": ranked, "profiles": [r["profile"] for r in ranked[:cfg["shortlist_size"]]]}


def choose_winner(frames, cfg):
    scores = {name: pooled(frame, cfg["tuning_dates"]) for name, frame in frames.items()}
    baseline = scores["baseline"]
    best = min((n for n in scores if n != "baseline"), key=lambda n: (scores[n], n))
    winner = best if baseline-scores[best] >= cfg["minimum_tuning_gain"] else "baseline"
    return {"profile": winner, "tuning_dates": cfg["tuning_dates"], "mean3_RMSLE": scores,
            "minimum_tuning_gain": cfg["minimum_tuning_gain"],
            "confirmation_used_for_selection": False}


def freeze(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Frozen decision differs: {path}")
    atomic_json(path, value)


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if str(path).endswith(".gz") else None)
    temporary.replace(path)


class Runner:
    def __init__(self, args, cfg, reference, data):
        self.args, self.cfg, self.reference, self.data = args, cfg, reference, data
        self.created = set()
        self.started = time.monotonic()
        self.settings = {p["name"]: {**cfg["base_settings"], **p["overrides"]} for p in cfg["profiles"]}
        if args.smoke:
            for settings in self.settings.values():
                settings.update(max_epochs=2, patience=1)

    def directory(self, name, date, seed, testing=False):
        return self.args.out_dir / ("refit_jobs" if testing else "jobs") / name / date / str(seed)

    def completed(self, name, date, seed, testing=False):
        directory = self.directory(name,date,seed,testing)
        path = directory / "result.json"
        if not path.exists():
            return None
        result = json.loads(path.read_text())
        spec = {"profile": name,"date": date,"seed": seed,"testing": testing,"settings": self.settings[name]}
        if result["job"] != spec:
            raise ValueError("Completed job parameters differ")
        for n, sha in result["sha256"].items():
            if digest(directory/n) != sha:
                raise ValueError(f"Corrupt completed artifact: {directory/n}")
        return result

    def control(self, date):
        ix = self.data.valid(date)
        frame = self.reference.frame(date).loc[self.data.objects.iloc[ix].Id].copy().reset_index(drop=True)
        frame['purchase_days_180'] = np.rint(np.expm1(self.data.base.features[ix,BASE_COLUMNS.index('n_180')])).astype(int)
        write_csv(self.args.out_dir / "predictions/baseline" / f"{date}.csv.gz", frame)
        return frame

    def execute(self, name, date, seed, testing=False):
        old = self.completed(name,date,seed,testing)
        if old:
            if not self.args.resume and (name,date,seed,testing) not in self.created:
                raise FileExistsError("Completed job exists; use --resume")
            return old
        if name == "baseline" and not testing:
            raise ValueError("Validation baseline must be reused, never retrained")
        limit = getattr(self.args,'max_minutes',0)
        if limit and time.monotonic()-self.started >= 60*limit:
            raise BudgetReached('Time budget reached between complete seed jobs')
        directory = self.directory(name,date,seed,testing)
        directory.mkdir(parents=True, exist_ok=True)
        ix, ti, iv, inner = temporal_split(self.data,date)
        vi = self.data.valid(date,testing)
        if not len(vi):
            raise ValueError("Empty validation/test population")
        obj = self.data.objects
        settings = self.settings[name]
        print(f"{name} {date} seed {seed}: selection on {inner.date()}",flush=True)
        start = time.monotonic()
        selector = Neural("mlp","hurdle_joint",settings,seed,self.args.device,self.args.threads)
        selector.fit(self.data.matrix(ti), obj.iloc[ti].spend_30d.to_numpy(float), row_weights(obj,ti,"D2"),
                     validation=(self.data.matrix(iv),obj.iloc[iv].spend_30d.to_numpy(float)))
        epoch = selector.best_epoch
        write_csv(directory/"selection.csv",pd.DataFrame(selector.history))
        del selector
        model = Neural("mlp","hurdle_joint",settings,seed,self.args.device,self.args.threads)
        model.fit(self.data.matrix(ix),obj.iloc[ix].spend_30d.to_numpy(float),row_weights(obj,ix,"D2"),epochs=epoch)
        write_csv(directory/"refit.csv",pd.DataFrame(model.history))
        model.save(directory/"model.pt")
        train_seconds = time.monotonic()-start
        start = time.monotonic()
        z = model.predict(self.data.matrix(vi))
        infer_seconds = time.monotonic()-start
        del model
        if not np.isfinite(z).all() or (z < 0).any():
            raise ValueError("Invalid log forecast")
        frame = obj.iloc[vi][["Id","customer_id"]].copy().reset_index(drop=True)
        frame["candidate"],frame["date"],frame["z"] = name,date,z
        frame['purchase_days_180'] = np.rint(np.expm1(self.data.base.features[vi,BASE_COLUMNS.index('n_180')])).astype(int)
        if testing:
            money = np.maximum(np.expm1(z),0)
            if not np.isfinite(money).all():
                raise ValueError("Non-finite money forecast")
            np.savez_compressed(directory/"predictions.npz",Id=frame.Id.to_numpy(str),
                                customer_id=frame.customer_id.to_numpy(str),z=z,spend_30d=money)
        else:
            control = self.control(date).set_index("Id").loc[frame.Id]
            frame["target_z"] = np.log1p(obj.iloc[vi].spend_30d.to_numpy(float))
            frame["reference_z"] = control.z.to_numpy()
            frame["delta_squared_log_error"] = (frame.z-frame.target_z)**2-(frame.reference_z-frame.target_z)**2
            write_csv(directory/"predictions.csv.gz",frame)
        result = {"job":{"profile":name,"date":date,"seed":seed,"testing":testing,"settings":settings},
                  "selected_epoch":epoch,"inner_date":str(inner.date()),"n_train":len(ix),"n_inner":len(ti),
                  "max_target_end_train":str(obj.iloc[ix].target_end.max()),
                  "max_target_end_inner":str(obj.iloc[ti].target_end.max()),
                  "n_validation":len(vi),"train_seconds":train_seconds,"inference_seconds":infer_seconds,
                  "sha256":{p.name:digest(p) for p in directory.iterdir() if p.is_file() and p.name!='result.json'}}
        if not testing:
            result['RMSLE'] = float(np.sqrt(np.mean((frame.z-frame.target_z)**2)))
        atomic_json(directory/"result.json",result)
        self.created.add((name,date,seed,testing))
        score = 'test' if testing else f"RMSLE={result['RMSLE']:.6f}"
        print(f"Completed {name} {date} seed {seed}: {score}, epoch={epoch}, fit={train_seconds:.1f}s",flush=True)
        return result

    def seed_frame(self,name,date,seed):
        if self.completed(name,date,seed) is None:
            raise ValueError("Required screening job is unfinished")
        return pd.read_csv(self.directory(name,date,seed)/"predictions.csv.gz",dtype={"Id":str,"customer_id":str})

    def mean3(self,name,date):
        if name == "baseline":
            return self.control(date)
        frames = [self.seed_frame(name,date,seed) for seed in self.cfg["seeds"]]
        frame = frames[0].copy()
        for other in frames[1:]:
            if not np.array_equal(other.Id,frame.Id):
                raise ValueError("Seed predictions are not aligned")
        frame["z"] = np.mean([a.z.to_numpy() for a in frames],axis=0)
        frame["delta_squared_log_error"] = (frame.z-frame.target_z)**2-(frame.reference_z-frame.target_z)**2
        write_csv(self.args.out_dir/"predictions"/name/f"{date}.csv.gz",frame)
        return frame

    def screen(self):
        frames, signatures = {}, {}
        for name in self.settings:
            if name == "baseline":
                continue
            frames[name] = []
            for date in self.cfg["tuning_dates"]:
                self.execute(name,date,self.cfg["screen_seed"])
                frames[name].append(self.seed_frame(name,date,self.cfg["screen_seed"]))
                p=self.directory(name,date,self.cfg["screen_seed"])/"result.json"
                signatures[str(p.relative_to(self.args.out_dir))]=digest(p)
            frames[name] = pd.concat(frames[name],ignore_index=True)
        decision = choose_shortlist(frames,self.cfg)
        decision["job_result_sha256"] = signatures
        freeze(self.args.out_dir/"shortlist.json",decision)
        write_csv(self.args.out_dir/"screen_scores.csv",pd.DataFrame(decision["ranking"]))
        return decision

    def confirm(self):
        # Recompute the shortlist from finished early jobs without fitting anything.
        path=self.args.out_dir/"shortlist.json"
        if not path.exists():
            raise ValueError("Run seed42 screening first")
        old=json.loads(path.read_text())
        frames={name:pd.concat([self.seed_frame(name,date,42) for date in self.cfg["tuning_dates"]],ignore_index=True)
                for name in self.settings if name!='baseline'}
        decision=choose_shortlist(frames,self.cfg)
        if decision != {k:v for k,v in old.items() if k!='job_result_sha256'}:
            raise ValueError("Screening decision was changed")
        for name,sha in old['job_result_sha256'].items():
            if digest(self.args.out_dir/name)!=sha:
                raise ValueError("Screening artifact changed after shortlisting")
        means={"baseline":pd.concat([self.control(date) for date in self.cfg["tuning_dates"]],ignore_index=True)}
        for name in decision["profiles"]:
            for date in self.cfg["tuning_dates"]:
                for seed in self.cfg["seeds"]:
                    self.execute(name,date,seed)
            means[name]=pd.concat([self.mean3(name,date) for date in self.cfg["tuning_dates"]],ignore_index=True)
        winner=choose_winner(means,self.cfg)
        freeze(self.args.out_dir/"winner.json",winner)  # BEFORE any Sep/Oct prediction/score is read.
        name=winner['profile']
        for date in self.cfg['confirmation_dates']:
            if name!='baseline':
                for seed in self.cfg['seeds']:
                    self.execute(name,date,seed)
                self.mean3(name,date)
            self.control(date)
        return winner

    def report(self):
        from .report import report
        report(self.args.out_dir,self.cfg)

    def refit(self, profile):
        marker=self.args.out_dir/'winner.json'
        if not marker.exists() or not (self.args.out_dir/'confirmation.json').exists():
            raise ValueError("Finish tuning/confirmation first")
        winner=json.loads(marker.read_text())['profile']
        name=winner if profile=='winner' else profile
        if name not in {winner,'baseline'}:
            raise ValueError("Refit accepts the frozen winner or baseline")
        objects=self.data.objects
        date=str(objects.loc[objects.is_test,'as_of_date'].min().date())
        values=[]
        for seed in self.cfg['seeds']:
            self.execute(name,date,seed,testing=True)
            with np.load(self.directory(name,date,seed,True)/'predictions.npz',allow_pickle=False) as p:
                values.append({k:p[k] for k in p.files})
        for value in values[1:]:
            if not np.array_equal(value['Id'],values[0]['Id']):
                raise ValueError("Unaligned refit seeds")
        z=np.mean([a['z'] for a in values],axis=0)
        path=self.args.out_dir/'refit'/name
        path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/'predictions.npz',Id=values[0]['Id'],customer_id=values[0]['customer_id'],
                            z=z,spend_30d=np.maximum(np.expm1(z),0))
        atomic_json(path/'result.json',{'profile':name,'cutoff':date,'submission_created':False,
                                       'predictions_sha256':digest(path/'predictions.npz')})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','run','screen','confirm','report','refit','export'])
    p.add_argument('--config',type=Path,default=Path('configs/history_tuning.json'))
    p.add_argument('--reference',type=Path,default=Path('outputs/behavior-v1'))
    p.add_argument('--base-cache',type=Path)
    p.add_argument('--data-dir',type=Path,default=Path('data/raw'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs/history-tuning-v1'))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--profile',default='winner')
    p.add_argument('--max-minutes',type=float,default=90.,help='Stop between seed jobs; 0 disables the time limit')
    p.add_argument('--archive',type=Path,default=Path('history-tuning-results.tar.gz'))
    args=p.parse_args(argv)
    if args.max_minutes < 0:
        p.error('--max-minutes must be nonnegative')
    cfg=read_config(args.config)
    if args.command=='plan':
        print(json.dumps({'tuning_dates':cfg['tuning_dates'],'confirmation_dates':cfg['confirmation_dates'],
                          'profiles':cfg['profiles'],**budget(cfg)},indent=2))
        return
    if args.command=='export':
        from .report import export
        print(json.dumps(export(args.out_dir,args.archive),indent=2))
        return
    args.out_dir.mkdir(parents=True,exist_ok=True)
    with (args.out_dir/'runner.lock').open('w') as lock,threadpool_limits(limits=args.threads):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reference=FrozenRun(args.reference,args.data_dir,cfg['base_settings'])
        data=TuningData(reference,args.base_cache,args.smoke)
        if args.smoke:
            cfg['bootstrap_repeats']=20
        sources=[q for package in ('spend_baselines','spend_behavior','spend_tuning')
                 for q in sorted((Path(__file__).parents[1]/package).glob('*.py'))]
        identity={'config':cfg,'smoke':args.smoke,'device':args.device,'threads':args.threads,
                  'python':platform.python_version(),'libraries':{n:version(n) for n in
                     ('numpy','pandas','scipy','scikit-learn','torch')},
                  'source_sha256':{f'{q.parent.name}/{q.name}':digest(q) for q in sources},
                  'reference_manifest_sha256':hashlib.sha256(reference.manifest_bytes).hexdigest(),
                  'reference_prediction_hashes':reference.prediction_hashes,'cache':data.identity}
        manifest=args.out_dir/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())!=identity:
            raise ValueError("Run definition changed; use a new output directory")
        atomic_json(manifest,identity)
        runner=Runner(args,cfg,reference,data)
        finished=True
        try:
            if args.command in {'run','screen'}:
                runner.screen()
            if args.command in {'run','confirm'}:
                runner.confirm()
            if args.command=='refit':
                runner.refit(args.profile)
        except BudgetReached as error:
            finished=False
            print(f'{error}; rerun the same command with --resume',flush=True)
        if args.command!='refit':
            runner.report()
        atomic_json(args.out_dir/'status.json',{'phase':args.command,'phase_complete':finished,
                                               'elapsed_seconds':time.monotonic()-runner.started})
