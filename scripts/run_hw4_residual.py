"""Controlled residual/embedding HW4 screen on V100, with no lockbox or uploads."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.classical_analysis import analyze_predictions
from hw4_inspections.classical_features import select_feature_columns
from hw4_inspections.classical_models import SPECS
from hw4_inspections.extensions import build_context_features
from hw4_inspections.features import build_features
from hw4_inspections.neural import build_sequences
from hw4_inspections.residual_models import ResidualInspectionNet
from hw4_inspections.residual_preprocessing import ResidualTransform
from hw4_inspections.residual_training import COMMON_OPTIMIZER, infer, seed_all, train_epochs

QUARTERS = ["2023-07-01", "2023-10-01", "2024-01-01"]
KINDS = ["mlp_control", "resnet", "ple_resnet", "ple_visit_transformer"]
RAW_FILES = ["train.csv", "test_student.csv", "inspection_history.csv", "sample_submission.csv"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_development(raw, *, smoke=False, sequences=True):
    train = pd.read_csv(raw / "train.csv")
    train = train.loc[train.forecast_period <= QUARTERS[-1]].reset_index(drop=True)
    if train.row_id.duplicated().any():
        raise ValueError("Duplicate official row_id")
    if smoke:
        train = train.groupby("forecast_period", group_keys=False).head(32).reset_index(drop=True)
    rows = train.drop(columns="target")
    history = pd.read_csv(raw / "inspection_history.csv")
    x, cats = build_features(rows, history)
    x = pd.concat([x, build_context_features(rows, history)], axis=1)
    x = x[select_feature_columns(x, "compact_context")]
    cats = [c for c in cats if c in x]
    if sequences:
        events, valid, audit = build_sequences(rows, history)
    else:
        events = np.empty((len(rows), 0, 60), dtype=np.float32)
        valid = np.empty((len(rows), 0), dtype=bool)
        audit = {"status": "skipped", "reason": "No visit model selected"}
    return train, x, cats, events, valid, audit


def classical_reference(directory, train, input_sha256, quarters):
    """Read existing reference/compact OOF without refitting or loading pickle."""
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol.get("status") != "complete" or protocol.get("config", {}).get("smoke") or protocol.get("lockbox_scored") is not False:
        raise ValueError("Classical reference must be a completed full dev run")
    if protocol["input_sha256"] != input_sha256:
        raise ValueError("Classical and neural official data hashes differ")
    for relative in ["src/hw4_inspections/features.py", "src/hw4_inspections/extensions.py",
                     "src/hw4_inspections/classical_features.py", "src/hw4_inspections/classical_models.py"]:
        if protocol["source_sha256"].get(relative) != sha(ROOT / relative):
            raise ValueError("Classical feature sources differ")
    for name in ["cb_reference", "cb_compact"]:
        if protocol.get("specs", {}).get(name) != SPECS[name]:
            raise ValueError("Classical reference specifications differ from the frozen screen")
    sources, parts = {}, []
    official = train.loc[train.forecast_period.isin(quarters)].set_index("row_id")
    for model in ["cb_reference", "cb_compact"]:
        frames = []
        for seed in protocol["config"]["seeds"]:
            for quarter in quarters:
                job = directory / f"{model}_s{seed}_{quarter}"
                marker = json.loads((job / "complete.json").read_text())
                file = job / "predictions.csv"
                if marker["signature"] != protocol["signature"] or marker["artifacts"]["predictions.csv"] != sha(file):
                    raise ValueError("Classical job prediction hash/signature mismatch")
                sources[str(file.relative_to(directory))] = sha(file)
                frame = pd.read_csv(file, float_precision="round_trip")
                if frame.row_id.duplicated().any() or not frame.model.eq(model).all():
                    raise ValueError("Invalid classical OOF identities/model")
                if not frame.forecast_period.eq(quarter).all() or not frame.seed.eq(seed).all():
                    raise ValueError("Classical OOF quarter/seed mismatch")
                expected = official.loc[official.forecast_period.eq(quarter)]
                # Smoke compares a deterministic subset of previously full OOF.
                frame = frame.set_index("row_id").reindex(expected.index)
                for col in ["site_id", "forecast_period", "target"]:
                    if not np.array_equal(frame[col].to_numpy(), expected[col].to_numpy()):
                        raise ValueError(f"Classical OOF mismatches official {col}")
                p = frame.prediction.to_numpy(dtype=float)
                if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    raise ValueError("Invalid classical prediction probabilities")
                frames.append(frame.reset_index())
        combined = pd.concat(frames, ignore_index=True)
        # One complete prediction per seed/row is required before averaging.
        if not combined.groupby("row_id").size().eq(len(protocol["config"]["seeds"])).all():
            raise ValueError("Missing or repeated classical seeds")
        values = combined.groupby("row_id", sort=False).prediction.mean()
        part = official.reset_index()
        part["prediction"] = part.row_id.map(values)
        part["model"] = model
        parts.append(part)
    return pd.concat(parts, ignore_index=True), {"protocol_sha256": sha(directory / "protocol.json"),
                                                "job_predictions_sha256": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/hw4-residual-v100-01")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--models", choices=KINDS, nargs="+", default=KINDS)
    parser.add_argument("--quarters", choices=QUARTERS, nargs="+", default=QUARTERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--classical-results", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.max_epochs, args.patience, args.batch_size, args.threads, args.width, args.blocks, args.bins) < 1 or args.bins > 64 or args.bootstrap < 1:
        raise ValueError("Invalid dimensions, epoch counts, threads or bootstrap")
    if any(len(v) != len(set(v)) for v in [args.models, args.quarters, args.seeds]):
        raise ValueError("Duplicate model, quarter or seed")
    missing = [name for name in RAW_FILES if not (args.raw / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CSV in {args.raw.resolve()}: {missing}. Extract official ZIP directly into this folder or set --raw.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    started = time.monotonic()

    def log(event, **details):
        print(json.dumps({"event": event, "elapsed_seconds": round(time.monotonic() - started, 3),
                          **details}), flush=True)

    sources = [Path(__file__), *(ROOT / "src/hw4_inspections" / name for name in
               ["features.py", "extensions.py", "neural.py", "classical_features.py", "classical_models.py", "classical_analysis.py",
                "residual_preprocessing.py", "residual_models.py", "residual_training.py"])]
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    provenance = {"config": config, "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
                  "input_sha256": {name: sha(args.raw / name) for name in RAW_FILES},
                  "versions": {name: importlib.metadata.version(name) for name in
                               ["torch", "numpy", "pandas", "scikit-learn", "joblib"]},
                  "python": platform.python_version(), "machine": platform.machine(),
                  "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                  "optimizer": COMMON_OPTIMIZER, "attention_backend": "math_only",
                  "fixed_architecture": {"embedding_dim": 8, "dropout": .15, "visit_width": 64,
                                         "visit_layers": 2, "visit_heads": 4, "visit_ffn": 128,
                                         "code_embedding_dim": 16, "sequence_length": 16}}
    # Include all reused classical prediction file hashes in the run signature.
    log("features_start", device="cpu")
    prepared = prepare_development(args.raw, smoke=args.smoke, sequences="ple_visit_transformer" in args.models)
    train, x, cats, events, valid, sequence_audit = prepared
    log("features_complete", rows=len(train), columns=len(x.columns), sequence_audit=sequence_audit)
    baseline, reference_provenance = None, None
    if args.classical_results:
        baseline, reference_provenance = classical_reference(args.classical_results, train, provenance["input_sha256"], args.quarters)
    provenance["classical_reference"] = reference_provenance
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if not args.resume or old["signature"] != signature:
            raise ValueError("Use new --out, or --resume with unchanged code/data/config/environment/reference")
    protocol = {**provenance, "signature": signature, "features": list(x), "categorical": cats,
                "sequence_audit": sequence_audit, "status": "running", "lockbox_scored": False,
                "test_refits": 0, "kaggle_uploads": 0, "embeddings_supervised": True,
                "downstream_embeddings_require_temporal_cross_fitting": True,
                "epoch_selection": "previous-quarter inner AP, then fresh outer fit with selected epoch count"}
    write_json(protocol_path, protocol)
    if device.type == "cuda":
        probe = torch.ones(1, device=device)
        assert (probe + 1).item() == 2
        log("cuda_ready", gpu=torch.cuda.get_device_name(0), device=str(probe.device))
    records, predictions, reused = [], [], 0

    def tensor_data(indices, preprocessing):
        numeric, categorical, basis = preprocessing.transform(x.iloc[indices])
        return tuple(torch.as_tensor(a, device=device) for a in
                     [numeric, categorical, basis, events[indices], valid[indices]])

    def fit(kind, seed, indices, eval_indices, epochs, quarter, phase):
        seed_all(seed)
        preprocessing = ResidualTransform(args.bins).fit(x.iloc[indices], cats)
        data = tensor_data(indices, preprocessing)
        kwargs = {"n_numeric": len(preprocessing.numeric_names), "cardinalities": preprocessing.cardinalities,
                  "kind": kind, "bins": args.bins, "width": args.width, "blocks": args.blocks}
        model = ResidualInspectionNet(**kwargs).to(device)
        target = torch.as_tensor(train.iloc[indices].target.to_numpy(dtype=np.float32), device=device)
        inner_data = tensor_data(eval_indices, preprocessing) if eval_indices is not None else None
        inner_target = torch.as_tensor(train.iloc[eval_indices].target.to_numpy(dtype=np.float32), device=device) if eval_indices is not None else None
        def progress(event, **details):
            log(event, model=kind, seed=seed, quarter=quarter, phase=phase, **details)
        progress("fit_start", train_n=len(indices), max_epochs=epochs, device=str(next(model.parameters()).device))
        chosen, curve = train_epochs(model, data, target, epochs=epochs, batch_size=args.batch_size,
                                     seed=seed, eval_data=inner_data, eval_targets=inner_target,
                                     patience=args.patience, log=progress)
        return model, preprocessing, kwargs, chosen, curve

    for kind in args.models:
        for seed in args.seeds:
            for quarter in args.quarters:
                job = args.out / f"{kind}_s{seed}_{quarter}"
                job.mkdir(exist_ok=True)
                marker = job / "complete.json"
                if args.resume and marker.exists():
                    record = json.loads(marker.read_text())
                    if record["signature"] != signature or any(sha(job / p) != h for p, h in record["artifacts"].items()):
                        raise ValueError("Completed neural job changed")
                    fold = pd.read_csv(job / "predictions.csv", float_precision="round_trip")
                    reused += 1
                    log("job_reused", model=kind, seed=seed, quarter=quarter)
                else:
                    t0 = time.monotonic()
                    fit_idx = np.flatnonzero(train.forecast_period.lt(quarter))
                    outer_idx = np.flatnonzero(train.forecast_period.eq(quarter))
                    inner_quarter = train.iloc[fit_idx].forecast_period.max()
                    inner_fit = np.flatnonzero(train.forecast_period.lt(inner_quarter))
                    inner_eval = np.flatnonzero(train.forecast_period.eq(inner_quarter))
                    inner_model, _, _, epochs, inner_curve = fit(kind, seed, inner_fit, inner_eval,
                                                                 args.max_epochs, quarter, "inner_selection")
                    del inner_model
                    model, preprocessing, kwargs, _, curve = fit(kind, seed, fit_idx, None, epochs, quarter, "outer_refit")
                    outer_data = tensor_data(outer_idx, preprocessing)
                    p = infer(model, outer_data, args.batch_size)
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, job / "model.pt")
                    joblib.dump(preprocessing, job / "preprocessing.joblib")
                    restored_preprocessing = joblib.load(job / "preprocessing.joblib")
                    loaded = ResidualInspectionNet(**kwargs).to(device)
                    loaded.load_state_dict(torch.load(job / "model.pt", map_location=device, weights_only=True))
                    restored = infer(loaded, tensor_data(outer_idx, restored_preprocessing), args.batch_size)
                    if not np.array_equal(p, restored):
                        raise AssertionError("Neural saved model/preprocessing forward differs")
                    def embeddings(network, data):
                        network.eval()
                        values = []
                        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                            for start in range(0, len(outer_idx), args.batch_size):
                                values.append(network.representation(*(a[start:start + args.batch_size] for a in data)).cpu().numpy())
                        result = np.concatenate(values)
                        if result.shape != (len(outer_idx), network.representation_dim) or not np.isfinite(result).all():
                            raise FloatingPointError("Invalid saved validation embeddings")
                        return result
                    representation = embeddings(model, outer_data)
                    if not np.array_equal(representation, embeddings(loaded, tensor_data(outer_idx, restored_preprocessing))):
                        raise AssertionError("Saved model/preprocessing embedding forward differs")
                    np.savez_compressed(job / "validation_embeddings.npz", row_id=train.iloc[outer_idx].row_id.to_numpy(dtype=str),
                                        embedding=representation)
                    fold = train.iloc[outer_idx][["row_id", "site_id", "forecast_period", "target"]].copy()
                    fold["model"], fold["seed"], fold["prediction"] = kind, seed, p
                    fold.to_csv(job / "predictions.csv", index=False)
                    write_json(job / "curves.json", {"inner": inner_curve, "outer": curve})
                    record = {"signature": signature, "model": kind, "seed": seed, "quarter": quarter,
                              "inner_quarter": inner_quarter, "selected_epochs": int(epochs), "model_kwargs": kwargs,
                              "parameters": sum(v.numel() for v in model.parameters()),
                              "ap": float(average_precision_score(fold.target, p)), "loaded_forward_exact": True,
                              "loaded_embeddings_exact": True,
                              "seconds": time.monotonic() - t0,
                              "artifacts": {n: sha(job / n) for n in ["model.pt", "preprocessing.joblib", "predictions.csv",
                                                                     "curves.json", "validation_embeddings.npz"]}}
                    write_json(marker, record)
                    log("fit_complete", **{k: record[k] for k in ["model", "seed", "quarter", "ap", "parameters", "selected_epochs", "loaded_forward_exact", "seconds"]})
                    del model, loaded, outer_data
                records.append(record)
                predictions.append(fold)
    long = pd.concat(predictions, ignore_index=True)
    long.to_csv(args.out / "predictions_by_seed.csv", index=False)
    metrics = pd.DataFrame([{k: r[k] for k in ["model", "seed", "quarter", "ap", "selected_epochs", "parameters", "seconds", "loaded_forward_exact", "loaded_embeddings_exact"]} for r in records])
    metrics.to_csv(args.out / "fold_metrics.csv", index=False)
    keys = ["row_id", "site_id", "forecast_period", "target", "model"]
    averaged = long.groupby(keys, sort=False, as_index=False).prediction.mean()
    report_files = ["protocol.json", "predictions_by_seed.csv", "fold_metrics.csv", "summary.csv"]
    if baseline is not None:
        combined = pd.concat([baseline, averaged], ignore_index=True)
        combined["seen_site"] = False
        for q in args.quarters:
            mask = combined.forecast_period.eq(q)
            combined.loc[mask, "seen_site"] = combined.loc[mask, "site_id"].isin(train.loc[train.forecast_period.lt(q), "site_id"])
        burden = pd.Series(x.history_violation_mean.to_numpy(), index=train.row_id)
        combined["history_violation_mean"] = combined.row_id.map(burden)
        result = analyze_predictions(combined, args.bootstrap)
        for name in ["predictions", "quarter_scores", "summary", "diagnostics"]:
            result[name].to_csv(args.out / f"{name}.csv", index=False)
        write_json(args.out / "comparison.json", result["comparison"])
        summary = result["summary"]
        report_files += ["predictions.csv", "quarter_scores.csv", "diagnostics.csv", "comparison.json"]
    else:
        # Proper ensemble AP across seeds, rather than average seed metric.
        per_quarter = averaged.groupby(["model", "forecast_period"], sort=False).apply(
            lambda f: average_precision_score(f.target, f.prediction), include_groups=False).rename("ap").reset_index()
        per_quarter.to_csv(args.out / "quarter_scores.csv", index=False)
        summary = per_quarter.groupby("model", as_index=False).agg(mean_ap=("ap", "mean"))
        summary.to_csv(args.out / "summary.csv", index=False)
        report_files.append("quarter_scores.csv")
    protocol.update(status="complete", completed_jobs=len(records), reused_jobs=reused,
                    elapsed_seconds=time.monotonic() - started)
    write_json(protocol_path, protocol)
    with tarfile.open(args.out / "return_results.tar.gz", "w:gz") as archive:
        for name in report_files:
            archive.add(args.out / name, arcname=name)
    print(summary.to_string(index=False), flush=True)
    log("complete", jobs=len(records), reused=reused, results=str(args.out / "return_results.tar.gz"),
        lockbox_scored=False, test_refits=0, kaggle_uploads=0)


if __name__ == "__main__":
    main()
