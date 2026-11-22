"""Post-Public exploratory MAP ranking versus matched logistic XGBoost.

Never scores the previously consumed 2024Q2/Q3 diagnostic. A quarter is a
ranking query; preprocessing and checkpoint selection use strictly past data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_hw4_final_research as trusted
from hw4_inspections.deep_boosting import FORBIDDEN_COLUMNS
from hw4_inspections.next_round import rank_scores

GRID = [100, 300, 600, 1000]


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def recipes():
    return {f"{objective}_{view}_d{depth}": {
        "objective": objective, "view": view, "depth": depth,
    } for objective in ["map", "logistic"]
        for view in ["full", "no_codes"] for depth in [2, 4]}


def prepare(frame, fit, pred, columns, cats):
    selected_cats = [c for c in cats if c in columns]
    fitting = frame.iloc[fit][columns].copy()
    prediction = frame.iloc[pred][columns].copy()
    for column in columns:
        if column in selected_cats:
            vocabulary = sorted(fitting[column].astype("string").dropna().unique())
            for part in [fitting, prediction]:
                part[column] = (pd.Categorical(part[column].astype("string"), categories=vocabulary)
                                if vocabulary else np.nan)
        else:
            for part in [fitting, prediction]:
                part[column] = part[column].astype(float).replace([np.inf, -np.inf], np.nan)
    return fitting, prediction


def fit_predict(xgb, recipe, frame, train, fit, pred, cats, count, seed):
    # Sorting is essential: native ranking requires contiguous query groups.
    fit = np.asarray(fit)
    order = np.argsort(train.iloc[fit].forecast_period.to_numpy(), kind="stable")
    fit = fit[order]
    columns = [c for c in frame if recipe["view"] == "full" or not c.startswith("code")]
    assert not FORBIDDEN_COLUMNS.intersection(columns)
    xfit, xpred = prepare(frame, fit, pred, columns, cats)
    dfit = xgb.DMatrix(xfit, label=train.target.to_numpy()[fit], enable_categorical=True, nthread=2)
    if recipe["objective"] == "map":
        qid = pd.factorize(train.iloc[fit].forecast_period, sort=True)[0].astype(np.uint32)
        assert np.all(qid[1:] >= qid[:-1])
        dfit.set_uint_info("qid", qid)
    dpred = xgb.DMatrix(xpred, enable_categorical=True, nthread=2)
    params = dict(objective="rank:map" if recipe["objective"] == "map" else "binary:logistic",
                  tree_method="hist", device="cpu", nthread=2, seed=seed,
                  max_depth=recipe["depth"], eta=.03, min_child_weight=1.,
                  reg_lambda=3., subsample=.8, colsample_bytree=.8,
                  max_bin=256, verbosity=0)
    if recipe["objective"] == "map":
        params.update(lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=4)
    with threadpool_limits(limits=2):
        model = xgb.train(params, dfit, num_boost_round=count)
        predictions = {n: model.predict(dpred, iteration_range=(0, n), output_margin=True).astype(float)
                       for n in GRID if n <= count}
        predictions[count] = model.predict(dpred, output_margin=True).astype(float)
    assert all(np.isfinite(v).all() for v in predictions.values())
    return model, predictions, dpred, columns


def main():
    parser = argparse.ArgumentParser()
    for flag in ["raw", "feature-cache", "baseline-cv", "context-test", "out"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.threads = 2
    args.bootstrap = 0
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Use a new empty output directory; no silent overwrite or rerun")
    import xgboost as xgb
    if xgb.__version__ != "3.4.1":
        raise ValueError("Requires already installed xgboost3.4.1; do not upgrade the environment")
    pool = recipes()
    plan = trusted.provenance(args, pool)
    plan.update(experiment="Post-Public MAP ranking vs matched logistic", recipes=pool,
                source_sha256_new=trusted.sha(Path(__file__)), grid=GRID,
                ranking_queries="forecast quarter, fit rows sorted before nativeDMatrix",
                selection="Each objective independently selects recipe+count on previous inner quarter sklearnAP; all24inner fits before6outer fits. Final one recipe by mean previous-inner AP, median count.",
                limits={"inner_fits": 24, "outer_fits": 6, "full_fits": 2},
                fixed_variants=["raw", "50%within-quarter fractionalrankC+50%rankcandidate"],
                full_gate="positive mean delta vs context and >=2/3 nonnegative; no recent scoring",
                interpretation="Exploratory follow-up after seeing11/12Public; heavily reused old quarters, NOT independent confirmation",
                new_recent_metrics=False,
                sources=["https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html"])
    write(args.out / "freeze.json", plan)
    started = time.monotonic()
    def log(event, **detail):
        print(json.dumps(dict(event=event, elapsed_seconds=round(time.monotonic()-started, 3), **detail)), flush=True)
    if args.smoke:
        # Native real-library check for grouping, prefix prediction and exact reload.
        x = pd.DataFrame({"a": [0., 1., 2., 3., 4., 5., np.nan, 7.]*2})
        t = pd.DataFrame({"forecast_period": ["2022-01-01"]*8+["2022-04-01"]*8,
                          "target": [0, 1, 0, 1, 0, 1, 0, 1]*2})
        model, values, data, _ = fit_predict(xgb, pool["map_full_d2"], x, t, np.arange(16), np.arange(16), [], 5, 42)
        path = args.out / "smoke.ubj"; model.save_model(path)
        reloaded = xgb.Booster();reloaded.load_model(path)
        assert np.array_equal(values[5], reloaded.predict(data, output_margin=True).astype(float))
        write(args.out / "complete.json", {"smoke": True, "native_rank_fit": True, "reload_exact": True})
        return
    cache = joblib.load(args.feature_cache)
    train, test, sample, views, cats = trusted.validate_cache(cache, args)
    frame = views["base"]
    meta = pd.concat([train.loc[train.forecast_period.eq(q), list(trusted.REQUIRED)]
                      for q in trusted.QUARTERS], ignore_index=True)
    anchor, anchor_test = trusted.align_anchors(meta, test, args)
    inner = {name: {} for name in pool}
    models_dir = args.out / "models";models_dir.mkdir()
    new_fits = 0
    for name, recipe in pool.items():
        for q in trusted.QUARTERS:
            iq = (pd.Period(q, freq="Q")-1).start_time.strftime("%Y-%m-%d")
            fi = np.flatnonzero(train.forecast_period.lt(iq))
            vi = np.flatnonzero(train.forecast_period.eq(iq))
            log("inner_start", recipe=name, outer=q, inner=iq)
            model, values, _, _ = fit_predict(xgb, recipe, frame, train, fi, vi, cats, GRID[-1], 42)
            new_fits += 1
            scores = {n: float(average_precision_score(train.iloc[vi].target, v)) for n, v in values.items()}
            count = min(scores, key=lambda n: (-scores[n], n))
            inner[name][q] = {"inner_quarter": iq, "count": count, "ap": scores[count],
                              "curve": [{"count": n, "ap": scores[n]} for n in sorted(scores)]}
            log("inner_complete", recipe=name, outer=q, count=count, ap=scores[count])
    write(args.out / "inner-curves.json", inner)
    choices = {}
    for objective in ["map", "logistic"]:
        names = [name for name, recipe in pool.items() if recipe["objective"] == objective]
        choices[objective] = {q: min(names, key=lambda name: (-inner[name][q]["ap"], name))
                              for q in trusted.QUARTERS}
        final = min(names, key=lambda name: (-np.mean([inner[name][q]["ap"] for q in trusted.QUARTERS]), name))
        choices[objective]["full"] = final
        choices[objective]["full_count"] = int(np.median([inner[final][q]["count"] for q in trusted.QUARTERS]))
    write(args.out / "inner-only-selection.json", choices)
    vectors = {"context_3seed": anchor}
    for objective in ["map", "logistic"]:
        chunks = []
        for q in trusted.QUARTERS:
            name = choices[objective][q]; recipe = pool[name]; count = inner[name][q]["count"]
            fi = np.flatnonzero(train.forecast_period.lt(q));vi = np.flatnonzero(train.forecast_period.eq(q))
            log("outer_start", objective=objective, quarter=q, recipe=name, count=count)
            model, values, dpred, _ = fit_predict(xgb, recipe, frame, train, fi, vi, cats, count, 42)
            new_fits += 1
            path = models_dir / f"{objective}__{q}.ubj";model.save_model(path)
            other = xgb.Booster();other.load_model(path)
            assert np.array_equal(values[count], other.predict(dpred, output_margin=True).astype(float))
            chunks.append(values[count]);log("outer_complete", objective=objective, quarter=q, reload_exact=True)
        raw = np.concatenate(chunks); vectors[objective] = raw
        mix = np.empty(len(meta))
        for q in trusted.QUARTERS:
            mask = meta.forecast_period.eq(q).to_numpy()
            mix[mask] = .5*rank_scores(anchor[mask])+.5*rank_scores(raw[mask])
        vectors[objective+"_context_rank50"] = mix
    rows = []
    for q in trusted.QUARTERS:
        mask = meta.forecast_period.eq(q).to_numpy()
        for name, v in vectors.items():
            rows.append({"quarter": q, "model": name, "ap": float(average_precision_score(meta.target[mask], v[mask]))})
    metrics = pd.DataFrame(rows);metrics.to_csv(args.out / "quarter-metrics.csv", index=False)
    table = metrics.pivot(index="quarter", columns="model", values="ap")
    summary = []
    for name in table:
        delta = table[name]-table.context_3seed
        summary.append({"model": name, "mean_ap": float(table[name].mean()),
                        "delta_vs_context": float(delta.mean()), "nonnegative_quarters": int((delta>=0).sum()),
                        "worst_delta": float(delta.min()),
                        "eligible": name!="context_3seed" and bool(delta.mean()>0 and (delta>=0).sum()>=2)})
    summary = pd.DataFrame(summary).sort_values("mean_ap", ascending=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    pd.concat([meta, pd.DataFrame(vectors)], axis=1).to_csv(args.out / "old-oof.csv", index=False)
    exports = []
    for objective in ["map", "logistic"]:
        eligible = summary[(summary.model.isin([objective, objective+"_context_rank50"])) & summary.eligible]
        if eligible.empty:
            continue
        chosen = eligible.iloc[0].model
        name = choices[objective]["full"];count=choices[objective]["full_count"]
        model, values, dpred, _ = fit_predict(xgb, pool[name], frame, train,
            np.arange(len(train)), np.arange(len(train), len(frame)), cats, count, 42)
        new_fits += 1
        path=models_dir/f"{objective}__full.ubj";model.save_model(path)
        other=xgb.Booster();other.load_model(path)
        assert np.array_equal(values[count],other.predict(dpred,output_margin=True).astype(float))
        raw = values[count]
        pred = (.5*rank_scores(anchor_test)+.5*rank_scores(raw)
                if chosen.endswith("rank50") else expit(raw))
        exported = trusted.validate_submission(test, sample, pred, len(test))
        output = args.out / f"submission_postpublic_{chosen}.csv";exported.to_csv(output,index=False)
        exports.append({"file":output.name,"model":chosen,"sha256":trusted.sha(output),"rows":len(exported)})
    write(args.out / "complete.json", {"new_fits":new_fits,"inner_fits":24,"outer_fits":6,
          "full_fits":new_fits-30,"recent_metrics":0,"kaggle_uploads":0,"exports":exports,
          "elapsed_seconds":time.monotonic()-started,"source_sha256":trusted.sha(Path(__file__))})
    print(summary.to_string(index=False),flush=True);log("complete",new_fits=new_fits,exports=len(exports))


if __name__ == "__main__":
    main()
