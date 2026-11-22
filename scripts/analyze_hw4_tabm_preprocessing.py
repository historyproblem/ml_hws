"""Validate seven GPU files; gate neural mixtures on old development only."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_hw4_tabm_preprocessing_v100 as producer
from hw4_inspections.delivery import validate_submission
from hw4_inspections.next_round import evaluate_candidates

REFERENCE = "previous07"
ANCHORS = {
    "baseline_cv": "0096456a838264be615680be4eead0ed8b449a4182d9f20acd519937e7cc1f7c",
    "context_test": "636f8cbe87367b6df0bae6d9a1d79a57de863bc78a21bc2296f5c2ff85970634",
    "additive_oof": "58755d1e12123d1dd18553ac45b5075b3b503cf3e1e5b5d4dd97ce526fc5c9a0",
    "additive_test": "ef638b33e25e391f943c8a3758a0ff27618b23467b01ae0a5d227aa5d9b4d0f0",
}


def read_csv(path_or_bytes):
    source = BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    return pd.read_csv(source, float_precision="round_trip")


def archive_members(path):
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if (len(members) != len(producer.RETURN_FILES) or {m.name for m in members} != set(producer.RETURN_FILES)
                or any(not m.isfile() or m.size > 8_000_000 for m in members)
                or sum(m.size for m in members) > 20_000_000):
            raise ValueError("Unexpected, duplicate, linked or oversized archive members")
        return {m.name: archive.extractfile(m).read() for m in members}


def scores(values, n, fp32=False):
    value = np.asarray(values, dtype=np.float64)
    if value.shape != (n,) or not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("Invalid row-aligned probabilities")
    if fp32 and not np.array_equal(value, value.astype(np.float32).astype(np.float64)):
        raise ValueError("Individual neural scores must be exact float32 expanded to float64")
    return value


def verify(args):
    files = archive_members(args.archive)
    protocol = json.loads(files["protocol.json"])
    completion = json.loads(files["completion.json"])
    signed = {k: v for k, v in protocol.items() if k != "signature"}
    if producer.signature(signed) != protocol["signature"] or completion["signature"] != protocol["signature"]:
        raise ValueError("Broken GPU protocol signature")
    if (protocol["format"] != "hw4-tabm-preprocessing-v100-1" or protocol["cases"] != producer.CASES
            or protocol["seeds"] != list(producer.SEEDS) or protocol["old_quarters"] != list(producer.QUARTERS)
            or protocol["new_2024q2_q3_metrics"] or protocol["external_pretrained_weights"]
            or protocol["kaggle_uploads"] or completion["new_2024q2_q3_metrics"] or completion["kaggle_uploads"]
            or completion["status"] != "complete" or completion["jobs"] != 63 or completion["expected_jobs"] != 63
            or completion["recent_scores_computed"] != 0 or protocol["recent_quarters"] != list(producer.RECENT)):
        raise ValueError("Unexpected neural protocol, folds or completion")
    if not args.smoke and (protocol["smoke"] or protocol["device"] != "cuda"
            or protocol["max_epochs"] != 60 or protocol["patience"] != 10
            or protocol["architecture"] != dict(k=32, width=256, blocks=3, bins=16, embedding_dim=8, dropout=.1, batch_size=256)):
        raise ValueError("Expected real frozen CUDA architecture and epoch policy")
    if args.smoke != protocol["smoke"]:
        raise ValueError("Cannot confuse synthetic and real GPU output")
    if protocol["versions"]["tabm"] != "0.0.3" or protocol["versions"]["rtdl-num-embeddings"] != "0.0.12":
        raise ValueError("Unexpected upstream neural package versions")
    if protocol["optimizer"] != producer.OPTIMIZER:
        raise ValueError("Changed optimizer")
    expected_sources = {"src/hw4_inspections/" + n for n in producer.SOURCES} | {"scripts/run_hw4_tabm_preprocessing_v100.py"}
    if set(protocol["source_sha256"]) != expected_sources:
        raise ValueError("Unexpected source signature set")
    for name, digest in protocol["source_sha256"].items():
        if producer.sha(ROOT / name) != digest:
            raise ValueError(f"GPU source differs: {name}")
    for name, digest in protocol["input_sha256"].items():
        if producer.sha(args.raw / name) != digest:
            raise ValueError("GPU and local official inputs differ")
    if set(protocol["input_sha256"]) != set(producer.RAW):
        raise ValueError("Missing official input signature")
    if set(completion["artifacts"]) != set(producer.RETURN_FILES) - {"completion.json"}:
        raise ValueError("Unexpected archive artifact signature set")
    for name, digest in completion["artifacts"].items():
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError("Changed archive artifact")
    if not args.smoke and any(producer.sha(getattr(args, name)) != digest for name, digest in ANCHORS.items()):
        raise ValueError("Local incumbent anchors changed")
    train, test, sample = [read_csv(args.raw / name) for name in ["train.csv", "test_student.csv", "sample_submission.csv"]]
    metadata = pd.concat([train.loc[train.forecast_period.eq(q), ["row_id", "site_id", "forecast_period", "target"]]
                          for q in producer.QUARTERS], ignore_index=True)
    old, forecast = read_csv(files["old_oof.csv"]), read_csv(files["test_predictions.csv"])
    recent = read_csv(files["recent_predictions.csv"])
    recent_meta = pd.concat([train.loc[train.forecast_period.eq(q), list(metadata)] for q in producer.RECENT], ignore_index=True)
    if not old.row_id.is_unique or not forecast.row_id.is_unique or not recent.row_id.is_unique:
        raise ValueError("Duplicate forecast IDs")
    pd.testing.assert_frame_equal(metadata, old[list(metadata)], check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(test[["row_id", "forecast_period"]], forecast[["row_id", "forecast_period"]], check_dtype=False)
    pd.testing.assert_frame_equal(recent_meta, recent[list(metadata)], check_dtype=False, check_exact=True)
    if completion["old_rows"] != len(old) or completion["test_rows"] != len(test):
        raise ValueError("Wrong official forecast sizes")
    expected = {f"{case}__s{seed}" for case in producer.CASES for seed in producer.SEEDS}
    expected |= {f"{case}__mean3" for case in producer.CASES}
    if (set(old) != set(metadata) | expected or set(forecast) != {"row_id", "forecast_period"} | expected
            or set(recent) != set(metadata) | expected or completion["recent_rows"] != len(recent)):
        raise ValueError("Unexpected prediction columns")
    for frame in [old, forecast, recent]:
        for case in producer.CASES:
            seeds = [scores(frame[f"{case}__s{seed}"], len(frame), fp32=True) for seed in producer.SEEDS]
            if not np.array_equal(scores(frame[f"{case}__mean3"], len(frame)), np.mean(seeds, axis=0)):
                raise ValueError("Changed three-seed probability mean")
    metrics, selections = read_csv(files["fold_metrics.csv"]), json.loads(files["epoch_selections.json"])
    keys = {tuple(row) for row in metrics[["case", "seed", "forecast_period"]].itertuples(index=False, name=None)}
    required_keys = {(c, s, q) for c in producer.CASES for s in producer.SEEDS for q in producer.QUARTERS}
    if len(metrics) != 27 or keys != required_keys or set(selections) != set(producer.CASES):
        raise ValueError("Incomplete neural folds")
    for record in metrics.to_dict("records"):
        mask = metadata.forecast_period.eq(record["forecast_period"])
        y = metadata.loc[mask, "target"]
        p = old.loc[mask, f"{record['case']}__s{record['seed']}"]
        if (record["n"] != len(y) or record["positive"] != int(y.sum())
                or abs(float(average_precision_score(y, p)) - record["ap"]) > 1e-12):
            raise ValueError("Reported old-fold AP/counts differ")
        count = selections[record["case"]]["inner_selected_epochs"][producer.QUARTERS.index(record["forecast_period"])]
        if count != record["epochs"] or not 1 <= count <= protocol["max_epochs"]:
            raise ValueError("Epoch selection differs across seeds")
    if any(item["full_epochs"] != int(np.median(item["inner_selected_epochs"])) for item in selections.values()):
        raise ValueError("Full epochs are not the old-inner median")
    return metadata, old, forecast, test, sample, protocol, recent


def run(args):
    meta, old, forecast, test, sample, protocol, recent = verify(args)
    incumbent = read_csv(args.additive_oof).set_index("row_id").loc[meta.row_id].reset_index()
    pd.testing.assert_frame_equal(meta, incumbent[list(meta)], check_dtype=False)
    anchor = scores(incumbent.additive_logit__blend75, len(meta))
    context = read_csv(args.baseline_cv).set_index("row_id").loc[meta.row_id]
    pd.testing.assert_frame_equal(meta.drop(columns="row_id").reset_index(drop=True),
                                  context[["site_id", "forecast_period", "target"]].reset_index(drop=True), check_dtype=False)
    context_values = scores(context.prediction, len(meta))
    inputs = {REFERENCE: anchor, "context04": context_values}
    choices = {}
    for case in producer.CASES:
        probability = old[f"{case}__mean3"].to_numpy()
        for variant, values in [("raw", probability), ("context75", .75 * context_values + .25 * probability),
                                ("additive75", .75 * anchor + .25 * probability)]:
            name = f"{case}__{variant}"
            inputs[name], choices[name] = values, (case, variant)
    result = evaluate_candidates(meta, inputs, reference=REFERENCE, bootstrap=args.bootstrap, seed=20261122)
    summary = result["summary"].copy()
    summary["eligible"] &= summary.worst_quarter_delta.ge(-.005)
    summary = summary.sort_values(["mean_ap", "model"], ascending=[False, True], kind="stable")
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    # Preserve already validated, deliberately unscored recent predictions for
    # the later single diagnostic of frozen finalists. This path computes no AP.
    recent.to_csv(args.out / "recent-unscored.csv", index=False)
    result["quarter_scores"].to_csv(args.out / "quarter-metrics.csv", index=False)
    pd.concat([meta, pd.DataFrame(inputs)], axis=1).to_csv(args.out / "old-oof.csv", index=False)
    eligible = summary.loc[summary.eligible & summary.model.isin(choices)]
    files = []
    if len(eligible):
        chosen = eligible.iloc[0]
        case, variant = choices[chosen.model]
        values = forecast[f"{case}__mean3"].to_numpy()
        if variant != "raw":
            parent = read_csv(args.context_test if variant == "context75" else args.additive_test).set_index("row_id").loc[test.row_id]
            values = .75 * scores(parent.target, len(test)) + .25 * values
        output = validate_submission(test, sample, values, len(test) if args.smoke else 2246)
        path = args.out / f"submission_{case}_{variant}.csv"
        output.to_csv(path, index=False)
        pd.testing.assert_frame_equal(output, read_csv(path), check_exact=True)
        files = [dict(file=path.name, sha256=producer.sha(path), rows=len(test), mean_ap=float(chosen.mean_ap),
                      delta_vs_previous07=float(chosen.delta_vs_reference), model=chosen.model,
                      public_ap=None, status="READY_NOT_SUBMITTED")]
    manifest = dict(status="PASS", source_archive_sha256=producer.sha(args.archive),
                    gpu_protocol_signature=protocol["signature"], smoke=args.smoke,
                    old_fold_metrics_recomputed=27, neural_three_seed_means_exact=True,
                    recent_predictions_verified_but_unscored=True,
                    reference=REFERENCE, files=files, new_2024q2_q3_metrics=False, kaggle_uploads=0,
                    interpretation="Reused old development, nominal intervals after adaptive selection; at most one of three fixed preprocessing recipes and fixed mixtures promoted.")
    producer.write_json(args.out / "verification.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["archive", "out", "raw", "baseline-cv", "context-test", "additive-oof", "additive-test"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
