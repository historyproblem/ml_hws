"""Controlled TabM preprocessing screen against the already evaluated 07 blend.

Three fixed fit-only preprocessing recipes, previous-quarter epoch selection,
fresh next-quarter refits, fixed probability mixtures, three-seed confirmation,
then at most one gated final forecast. No new 2024Q2/Q3 evaluation or upload.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_hw4_final_research as primary
from hw4_inspections.next_round import evaluate_candidates
from hw4_inspections.tabm_preprocessing import CASES, feature_view, fit_bounded
from hw4_inspections.tabm_training import OPTIMIZER, predict_bundle

REFERENCE = "additive75_incumbent"
ADDITIVE_OOF_SHA = "58755d1e12123d1dd18553ac45b5075b3b503cf3e1e5b5d4dd97ce526fc5c9a0"
ADDITIVE_TEST_SHA = "ef638b33e25e391f943c8a3758a0ff27618b23467b01ae0a5d227aa5d9b4d0f0"
sha, read_json, write_json = primary.sha, primary.read_json, primary.write_json
read_csv, write_csv = primary.read_csv, primary.write_csv


def provenance(args):
    pool = {name: dict(spec, family="tabm_preprocessing") for name, spec in CASES.items()}
    plan = primary.provenance(args, pool)
    names = ["tabm_preprocessing.py", "tabm_models.py", "tabm_training.py", "neural.py", "classical_features.py"]
    plan["source_sha256"].update({name: sha(ROOT / "src/hw4_inspections" / name) for name in names})
    plan["source_sha256"][Path(__file__).name] = sha(Path(__file__))
    extra = dict(old_additive_oof=sha(args.additive_oof), old_additive75_test=sha(args.additive_test))
    if not args.smoke and extra != dict(old_additive_oof=ADDITIVE_OOF_SHA, old_additive75_test=ADDITIVE_TEST_SHA):
        raise ValueError("Expected exact previously validated additive OOF and 07 test forecast")
    plan.update(
        additional_anchors=extra, torch_packages={n: metadata.version(n) for n in ["torch", "tabm", "rtdl-num-embeddings"]},
        neural=dict(k=2 if args.smoke else 32, width=8 if args.smoke else 256,
                    blocks=1 if args.smoke else 3, bins=4 if args.smoke else 16,
                    embedding_dim=2 if args.smoke else 8, dropout=.1, batch_size=256,
                    max_epochs=args.max_epochs, patience=args.patience, optimizer=OPTIMIZER,
                    mode="CPU, FP32, deterministic", numeric_quantiles=512, quantile_subsample=None,
                    missing_flags="binary unscaled; train-constant continuous coordinates fixed; categories fit-only"),
        seeds=[42, 17, 2026], selection="Best one eligible case/variant from three fixed preprocessing arms, seed42 old-dev mean; confirm all3 seeds, reconsider same case variants; no seed subset. Adaptive reused-dev selection.",
        variants=["raw", "75%context04+25%neural", "75%previous07+25%neural"],
        admission_reference=REFERENCE, admission="positive mean delta to07; >=2 nonnegative old quarters; worst>=-.005",
        final="Three seeds, median three previous-inner selected epochs for chosen preprocessing; at most one CSV",
    )
    return plan


def job(args, signature, case, seed, quarter, stage, train, frame, cats, epochs=None):
    spec = CASES[case]
    if stage == "inner":
        period = train.loc[train.forecast_period.lt(quarter), "forecast_period"].max()
    else:
        period = quarter
    fit = np.arange(len(train)) if stage == "full" else np.flatnonzero(train.forecast_period.lt(period))
    pred = np.arange(len(train), len(frame)) if stage == "full" else np.flatnonzero(train.forecast_period.eq(period))
    if not len(fit) or not len(pred):
        raise ValueError("Empty neural split")
    identity = dict(signature=signature, case=case, seed=seed, quarter=quarter, stage=stage,
                    actual_period=period, train_rows=len(fit), predict_rows=len(pred), fixed_epochs=epochs)
    directory = args.out / "jobs" / f"{case}__{quarter}__s{seed}__{stage}"
    if directory.exists():
        if not args.resume:
            raise ValueError("Existing neural job requires --resume")
        rec = read_json(directory / "complete.json")
        if (rec["identity"] != identity or not rec["reload_exact"]
                or set(rec["sha256"]) != {"model.joblib", "prediction.npy", "curve.json"}
                or any(sha(directory / name) != digest for name, digest in rec["sha256"].items())):
            raise ValueError("Changed neural checkpoint or split identity")
        values = np.load(directory / "prediction.npy", allow_pickle=False)
        print(json.dumps(dict(event="job_reused", **identity)), flush=True)
        return primary.probabilities(values, len(pred)), rec["selected_epochs"]
    started = time.monotonic()
    print(json.dumps(dict(event="job_start", **identity)), flush=True)
    def emit(event, **detail):
        if event in {"epoch_complete", "first_batch_complete", "fit_start"}:
            print(json.dumps(dict(event=event, case=case, seed=seed, quarter=quarter, stage=stage, **detail)), flush=True)
    kwargs = dict(k=2 if args.smoke else 32, width=8 if args.smoke else 256,
                  blocks=1 if args.smoke else 3, bins=4 if args.smoke else 16,
                  embedding_dim=2 if args.smoke else 8, seed=seed, device="cpu",
                  epochs=args.max_epochs if stage == "inner" else epochs,
                  patience=args.patience, batch_size=256, threads=args.threads, log=emit)
    if stage == "inner":
        kwargs.update(eval_frame=frame.iloc[pred], eval_target=train.target.iloc[pred])
    bundle = fit_bounded(frame.iloc[fit], train.target.iloc[fit], cats, mode=spec["mode"], **kwargs)
    values = predict_bundle(bundle, frame.iloc[pred])
    directory.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=directory.parent, prefix=".pending-neural-") as temporary:
        temp = Path(temporary) / "job"
        temp.mkdir()
        joblib.dump(bundle, temp / "model.joblib", compress=1)
        if not np.array_equal(values, predict_bundle(joblib.load(temp / "model.joblib"), frame.iloc[pred])):
            raise ValueError("Neural checkpoint forward differs")
        np.save(temp / "prediction.npy", values, allow_pickle=False)
        write_json(temp / "curve.json", bundle["curve"])
        rec = dict(identity=identity, selected_epochs=bundle["epochs"],
                   actual_trained_epochs=bundle["trained_epochs"], inner_best_ap=bundle["best_ap"],
                   reload_exact=True, sha256={name: sha(temp / name) for name in ["model.joblib", "prediction.npy", "curve.json"]},
                   seconds=time.monotonic() - started, new_q2_q3_metrics=False)
        write_json(temp / "complete.json", rec)
        os.rename(temp, directory)
    print(json.dumps(dict(event="job_complete", **identity, selected_epochs=bundle["epochs"], seconds=rec["seconds"])), flush=True)
    return primary.probabilities(values, len(pred)), rec["selected_epochs"]


def assess(args, stage, meta, anchor, context, values):
    predictions = {REFERENCE: anchor, "context_3seed": context}
    identity = {}
    for case, value in values.items():
        for variant, scores in [("raw", value), ("context75", .75 * context + .25 * value),
                                ("additive75", .75 * anchor + .25 * value)]:
            name = f"{case}__{variant}"
            predictions[name] = scores
            identity[name] = (case, variant)
    result = evaluate_candidates(meta, predictions, reference=REFERENCE, bootstrap=args.bootstrap, seed=20261122)
    summary = result["summary"].copy()
    summary["eligible"] &= summary.worst_quarter_delta.ge(-.005)
    summary["case"] = [identity.get(name, (None, None))[0] for name in summary.model]
    summary["variant"] = [identity.get(name, (None, None))[1] for name in summary.model]
    summary = summary.sort_values(["mean_ap", "model"], ascending=[False, True], kind="stable")
    eligible = summary.loc[summary.eligible & summary.case.notna()]
    fields = ["model", "case", "variant", "mean_ap", "delta_vs_reference"]
    choice = eligible.iloc[0][fields].to_dict() if len(eligible) else None
    write_csv(args.out / f"{stage}-summary.csv", summary)
    write_csv(args.out / f"{stage}-quarter-metrics.csv", result["quarter_scores"])
    write_csv(args.out / f"{stage}-oof.csv", pd.concat([meta, pd.DataFrame(predictions)], axis=1))
    write_json(args.out / f"{stage}-selection.json", choice)
    return choice


def run(args):
    if min(args.max_epochs, args.patience, args.threads) < 1 or args.threads > 2:
        raise ValueError("Positive limits and at most two CPU threads required")
    plan = provenance(args)
    plan["signature"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    signature = plan["signature"]
    freeze = args.out / "freeze.json"
    if freeze.exists():
        if not args.resume or read_json(freeze) != plan:
            raise ValueError("Changed neural inputs/source/recipe; use a new directory")
    else:
        if any(p.name not in {".lock", "run.log"} for p in args.out.iterdir()):
            raise ValueError("New neural run requires an empty output directory")
        write_json(freeze, plan)
    train, test, sample, views, cats = primary.validate_cache(joblib.load(args.feature_cache), args)
    meta = pd.concat([train.loc[train.forecast_period.eq(q), list(primary.REQUIRED)] for q in primary.QUARTERS], ignore_index=True)
    context, context_test = primary.align_anchors(meta, test, args)
    additional = read_csv(args.additive_oof).set_index("row_id").loc[meta.row_id].reset_index()
    pd.testing.assert_frame_equal(meta, additional[list(primary.REQUIRED)], check_dtype=False)
    anchor = primary.probabilities(additional.additive_logit__blend75, len(meta))
    reference_csv = read_csv(args.additive_test)
    reference_test = primary.validate_submission(reference_csv, test, reference_csv.target, len(test)).target.to_numpy()
    prepared = {name: feature_view(views["base"], cats, spec["view"]) for name, spec in CASES.items()}
    oof, counts = {}, {}
    for case, (frame, categories) in prepared.items():
        values, counts[case] = [], []
        for quarter in primary.QUARTERS:
            _, epochs = job(args, signature, case, 42, quarter, "inner", train, frame, categories)
            value, _ = job(args, signature, case, 42, quarter, "outer", train, frame, categories, epochs)
            values.append(value)
            counts[case].append(epochs)
        oof[case] = np.concatenate(values)
    selected = assess(args, "screen", meta, anchor, context, oof)
    files = []
    if selected:
        case = selected["case"]
        frame, categories = prepared[case]
        seeds = [oof[case]]
        for seed in [17, 2026]:
            values = [job(args, signature, case, seed, q, "outer", train, frame, categories, n)[0]
                      for q, n in zip(primary.QUARTERS, counts[case])]
            seeds.append(np.concatenate(values))
        confirmed = assess(args, "confirmed", meta, anchor, context, {case: np.mean(seeds, axis=0)})
        if confirmed:
            epochs = int(np.median(counts[case]))
            forecast = np.mean([job(args, signature, case, seed, "full", "full", train, frame, categories, epochs)[0]
                                for seed in [42, 17, 2026]], axis=0)
            variant = confirmed["variant"]
            if variant == "context75":
                forecast = .75 * context_test + .25 * forecast
            elif variant == "additive75":
                forecast = .75 * reference_test + .25 * forecast
            path = args.out / f"submission_{case}_{variant}.csv"
            write_csv(path, primary.validate_submission(test, sample, forecast, len(test) if args.smoke else 2246))
            files = [dict(case=case, variant=variant, sha256=sha(path), csv=path.name,
                          rows=len(test), epochs=epochs, mean_ap=confirmed["mean_ap"],
                          delta_vs_previous07=confirmed["delta_vs_reference"])]
    manifests = [p for p in args.out.iterdir() if p.suffix in {".csv", ".json"} and p.name != "forecast-manifest.json"]
    result = dict(complete=True, signature=signature, smoke=args.smoke, old_dev_only=True,
                  new_q2_q3_metrics=False, uploads=0, files=files, inner_selected_epochs=counts,
                  source_jobs=len(list((args.out / "jobs").glob("*/complete.json"))),
                  output_sha256={p.name: sha(p) for p in manifests})
    write_json(args.out / "forecast-manifest.json", result)
    print(json.dumps(dict(event="neural_preprocessing_complete", **result)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "out", "feature-cache", "baseline-cv", "context-test", "additive-oof", "additive-test"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
