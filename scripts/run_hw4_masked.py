"""Controlled masked-feature pretraining, supervised control and SSL64+CatBoost."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.extensions import build_context_features
from hw4_inspections.features import build_features
from hw4_inspections.masked import GroupMaskedNet, MaskedTransform
from hw4_inspections.masked_training import finetune, infer, pretrain, seed_all

QUARTERS = ["2023-07-01", "2023-10-01", "2024-01-01"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/hw4-masked-v100-01")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--variants", nargs="+", choices=["supervised", "masked"], default=["supervised", "masked"])
    parser.add_argument("--quarters", nargs="+", choices=QUARTERS, default=QUARTERS)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--mask-probability", type=float, default=.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--with-catboost", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.width % 4 or min(args.pretrain_epochs, args.max_epochs, args.patience, args.batch_size, args.layers, args.threads) < 1:
        raise ValueError("Invalid training dimensions or epoch counts")
    if not 0 < args.mask_probability < 1:
        raise ValueError("Mask probability must be between zero and one")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    if args.with_catboost:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("Install configs/hw4-masked-requirements.txt first") from exc
    started = time.monotonic()

    def log(event, **details):
        memory = round(torch.cuda.memory_allocated() / 1024**2, 2) if args.device == "cuda" else None
        print(json.dumps({"event": event, "elapsed_seconds": round(time.monotonic() - started, 3),
                          "cuda_allocated_mib": memory, **details}), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), *(ROOT / "src/hw4_inspections" / name for name in
                ["features.py", "extensions.py", "neural.py", "masked.py", "masked_training.py"])]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    provenance = {"config": config, "sources": {str(p.relative_to(ROOT)): sha(p) for p in sources},
                  "inputs": {p.name: sha(p) for p in sorted(args.raw.glob("*.csv"))},
                  "versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                               "sklearn": __import__("sklearn").__version__, "joblib": joblib.__version__},
                  "hardware": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"}
    if args.with_catboost:
        provenance["versions"]["catboost"] = __import__("catboost").__version__
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if not args.resume or old["signature"] != signature:
            raise ValueError("Use a new output directory, or --resume with unchanged code/data/config/environment")
    protocol = {"signature": signature, **provenance, "status": "running", "lockbox_scored": False,
                "pretraining_uses_targets": False, "pretraining_uses_validation_or_test_inputs": False,
                "catboost_embeddings": "Frozen before supervised fine-tuning; fit on earlier-quarter inputs only",
                "kaggle_uploads": 0}
    write_json(protocol_path, protocol)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if args.device == "cuda":
        probe = torch.ones(1, device=device)
        assert (probe + 1).item() == 2
        log("cuda_ready", gpu=torch.cuda.get_device_name(0), device=str(probe.device))
        del probe
    log("read_csv_start", device="cpu")
    train = pd.read_csv(args.raw / "train.csv")
    train = train.loc[train.forecast_period <= QUARTERS[-1]].reset_index(drop=True)
    if args.smoke:
        train = train.groupby("forecast_period", group_keys=False).head(32).reset_index(drop=True)
    history = pd.read_csv(args.raw / "inspection_history.csv")
    log("base_features_start", device="cpu")
    X, cats = build_features(train.drop(columns="target"), history)
    log("context_features_start", device="cpu")
    X = pd.concat([X, build_context_features(train.drop(columns="target"), history)], axis=1)
    del history
    log("features_complete", rows=len(X), columns=X.shape[1])
    y = train.target.to_numpy(dtype=np.float32)
    records, predictions, reused = [], [], 0

    def publish_tables():
        metrics = pd.DataFrame(records)
        metrics.to_csv(args.out / "fold_metrics.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(args.out / "predictions.csv", index=False)
        metrics.groupby("model", as_index=False).agg(mean_ap=("ap", "mean"), folds=("quarter", "size")).to_csv(
            args.out / "summary.csv", index=False)

    def fit_phase(variant, fit_idx, eval_idx, quarter, phase, epochs, folder):
        assert train.iloc[fit_idx].forecast_period.max() < train.iloc[eval_idx].forecast_period.min()
        preprocessing = MaskedTransform().fit(X.iloc[fit_idx], cats)
        fit_data = tuple(torch.as_tensor(a, device=device) for a in preprocessing.transform(X.iloc[fit_idx]))
        eval_data = tuple(torch.as_tensor(a, device=device) for a in preprocessing.transform(X.iloc[eval_idx]))
        model_args = {"n_numeric": len(preprocessing.numeric), "cardinalities": preprocessing.cardinalities,
                      "groups": preprocessing.groups, "width": args.width, "layers": args.layers}
        seed_all(args.seed)
        model = GroupMaskedNet(**model_args).to(device)
        if next(model.parameters()).device.type != device.type or any(a.device.type != device.type for a in (*fit_data, *eval_data)):
            raise RuntimeError("Model and tensors must use the requested device")

        def phase_log(event, **details):
            log(event, variant=variant, quarter=quarter, phase=phase, **details)

        phase_log("fit_start", device=str(next(model.parameters()).device), groups=len(preprocessing.groups),
                  max_fit_period=str(train.iloc[fit_idx].forecast_period.max()), fit_rows=len(fit_idx),
                  parameters=sum(p.numel() for p in model.parameters()))
        curves = {}
        frozen = None
        if variant == "masked":
            curves["pretraining"] = pretrain(model, fit_data, epochs=args.pretrain_epochs, batch_size=args.batch_size,
                mask_probability=args.mask_probability, seed=args.seed, log=phase_log)
            if phase == "outer":
                # Capture unsupervised weights/features BEFORE touching target.
                torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, folder / "pretrained.pt")
                frozen = (infer(model, fit_data, args.batch_size, embeddings=True),
                          infer(model, eval_data, args.batch_size, embeddings=True))
                restored_encoder = GroupMaskedNet(**model_args).to(device)
                restored_encoder.load_state_dict(torch.load(folder / "pretrained.pt", map_location=device, weights_only=True))
                np.testing.assert_array_equal(frozen[1], infer(restored_encoder, eval_data, args.batch_size, embeddings=True))
                del restored_encoder
                np.savez_compressed(folder / "frozen_ssl64.npz", train_row_id=train.iloc[fit_idx].row_id.to_numpy(dtype=str),
                                    valid_row_id=train.iloc[eval_idx].row_id.to_numpy(dtype=str),
                                    train_embedding=frozen[0], valid_embedding=frozen[1])
                phase_log("frozen_ssl64_saved", targets_used=False)
        target = torch.as_tensor(y[fit_idx], device=device)
        selected, curves["finetuning"] = finetune(model, fit_data, target, epochs=epochs, patience=args.patience,
            batch_size=args.batch_size, seed=args.seed, log=phase_log,
            eval_data=eval_data if phase == "inner" else None,
            eval_y=y[eval_idx] if phase == "inner" else None)
        write_json(folder / f"{phase}_curves.json", curves)
        return model, preprocessing, eval_data, selected, model_args, frozen

    for quarter in args.quarters:
        fit_idx = np.flatnonzero((train.forecast_period < quarter).to_numpy())
        valid_idx = np.flatnonzero((train.forecast_period == quarter).to_numpy())
        inner_quarter = train.iloc[fit_idx].forecast_period.max()
        inner_fit = np.flatnonzero((train.forecast_period < inner_quarter).to_numpy())
        inner_eval = np.flatnonzero((train.forecast_period == inner_quarter).to_numpy())
        for variant in args.variants:
            job_started = time.monotonic()
            folder = args.out / quarter / variant
            folder.mkdir(parents=True, exist_ok=True)
            done_path = folder / "job.json"
            if done_path.exists():
                done = json.loads(done_path.read_text())
                assert done["signature"] == signature
                for name, digest in done["sha256"].items():
                    if sha(folder / name) != digest:
                        raise ValueError(f"Completed job artifact changed: {folder / name}")
                records.extend(done["metrics"])
                predictions.append(pd.read_csv(folder / "predictions.csv"))
                reused += 1
                log("job_reused", quarter=quarter, variant=variant)
                publish_tables()
                continue
            _, _, _, epochs, _, _ = fit_phase(variant, inner_fit, inner_eval, quarter, "inner", args.max_epochs, folder)
            model, preprocessing, eval_data, _, model_args, frozen = fit_phase(
                variant, fit_idx, valid_idx, quarter, "outer", epochs, folder)
            p = infer(model, eval_data, args.batch_size)
            checkpoint = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(checkpoint, folder / "model.pt")
            joblib.dump(preprocessing, folder / "preprocessing.joblib")
            write_json(folder / "model_args.json", model_args)
            reloaded = GroupMaskedNet(**model_args).to(device)
            reloaded.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True))
            loaded_preprocessing = joblib.load(folder / "preprocessing.joblib")
            loaded_data = tuple(torch.as_tensor(a, device=device) for a in loaded_preprocessing.transform(X.iloc[valid_idx]))
            np.testing.assert_array_equal(p, infer(reloaded, loaded_data, args.batch_size))
            job_predictions, job_records = [], []

            def add_result(name, forecast, **extra):
                assert np.isfinite(forecast).all() and ((forecast >= 0) & (forecast <= 1)).all()
                row = {"model": name, "quarter": quarter, "seed": args.seed, "selected_epochs": epochs,
                       "ap": float(average_precision_score(y[valid_idx], forecast)), "train_rows": len(fit_idx),
                       "valid_rows": len(valid_idx), "loaded_forward_exact": True, **extra}
                job_records.append(row)
                pred = train.iloc[valid_idx][["row_id", "site_id", "forecast_period", "target"]].copy()
                pred["model"], pred["prediction"] = name, forecast
                job_predictions.append(pred)
                log("fold_complete", **row)

            add_result("group_" + variant, p, parameters=sum(a.numel() for a in model.parameters()))
            if variant == "masked" and args.with_catboost:
                trained, heldout = X.iloc[fit_idx].copy(), X.iloc[valid_idx].copy()
                for i in range(frozen[0].shape[1]):
                    trained[f"ssl64_{i:02d}"], heldout[f"ssl64_{i:02d}"] = frozen[0][:, i], frozen[1][:, i]
                cb = CatBoostClassifier(iterations=500, depth=5, learning_rate=.04, l2_leaf_reg=10,
                    random_seed=args.seed, thread_count=args.threads, verbose=False, allow_writing_files=False)
                log("catboost_ssl64_start", quarter=quarter, device="cpu", encoder_uses_targets=False)
                cb.fit(trained, y[fit_idx], cat_features=cats)
                prediction = cb.predict_proba(heldout)[:, 1]
                cb.save_model(str(folder / "catboost_ssl64.cbm"))
                restored = CatBoostClassifier()
                restored.load_model(str(folder / "catboost_ssl64.cbm"))
                np.testing.assert_array_equal(prediction, restored.predict_proba(heldout)[:, 1])
                add_result("catboost_ssl64", prediction, embedding_uses_targets=False)
            pd.concat(job_predictions, ignore_index=True).to_csv(folder / "predictions.csv", index=False)
            done = {"signature": signature, "metrics": job_records, "seconds": time.monotonic() - job_started,
                    "max_pretraining_period": str(train.iloc[fit_idx].forecast_period.max()) if variant == "masked" else None,
                    "sha256": {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != "job.json"}}
            write_json(done_path, done)
            records.extend(job_records)
            predictions.extend(job_predictions)
            publish_tables()
    protocol.update(status="complete", jobs_reused=reused, features=list(X), categorical=cats)
    write_json(protocol_path, protocol)
    log("run_complete", jobs_reused=reused, output=str(args.out))
    print(pd.read_csv(args.out / "summary.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
