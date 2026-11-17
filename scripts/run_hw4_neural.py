"""HW4 neural screen: nested temporal epoch selection, no lockbox or uploads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.extensions import build_context_features
from hw4_inspections.features import build_features
from hw4_inspections.neural import InspectionNet, TabularTransform, build_sequences

QUARTERS = ["2023-07-01", "2023-10-01", "2024-01-01"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data",
                        help="Directory containing the official CSV files (default: data/)")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/hw4-neural-20261117")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--models", nargs="+", choices=["mlp", "visit_transformer", "feature_transformer"],
                        default=["mlp", "visit_transformer"])
    parser.add_argument("--quarters", nargs="+", choices=QUARTERS, default=QUARTERS)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    if args.width % 4 or args.max_epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("Invalid width/epoch/batch/patience")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "protocol.json").exists():
        raise FileExistsError("Use a new output directory to preserve earlier runs")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    train = pd.read_csv(args.raw / "train.csv")
    train = train.loc[train.forecast_period <= QUARTERS[-1]].reset_index(drop=True)
    if args.smoke:
        train = train.groupby("forecast_period", group_keys=False).head(32).reset_index(drop=True)
    history = pd.read_csv(args.raw / "inspection_history.csv")
    rows = train.drop(columns="target")
    X, cats = build_features(rows, history)
    X = pd.concat([X, build_context_features(rows, history)], axis=1)
    events, valid_events, sequence_audit = build_sequences(rows, history)
    del history
    protocol = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "features": list(X), "categorical": cats, "sequence_audit": sequence_audit,
                "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
                    ROOT / "src/hw4_inspections/neural.py", ROOT / "src/hw4_inspections/features.py",
                    ROOT / "src/hw4_inspections/extensions.py"]},
                "input_sha256": {p.name: sha(p) for p in sorted(args.raw.glob("*.csv"))},
                "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .01},
                "epoch_selection": "previous quarter inner AP; refit from scratch on all earlier quarters",
                "outer_quarters": args.quarters, "lockbox_scored": False, "uploads": 0,
                "versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
                "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
                "status": "running"}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    device = torch.device(args.device)
    y = train.target.to_numpy(dtype=np.float32)
    metrics, predictions = [], []

    def prepare(indices, preprocessing):
        numeric, categorical = preprocessing.transform(X.iloc[indices])
        return tuple(torch.as_tensor(a, device=device) for a in
                     [numeric, categorical, events[indices], valid_events[indices]])

    def predict(model, data):
        model.eval()
        result = []
        with torch.no_grad():
            for start in range(0, len(data[0]), args.batch_size):
                result.append(model(*(a[start:start + args.batch_size] for a in data)).sigmoid().cpu().numpy())
        return np.concatenate(result)

    def fit(kind, fit_indices, eval_indices, epochs, select):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        preprocessing = TabularTransform().fit(X.iloc[fit_indices], cats)
        fit_data, eval_data = prepare(fit_indices, preprocessing), prepare(eval_indices, preprocessing)
        model = InspectionNet(fit_data[0].shape[1], preprocessing.cardinalities,
                              events.shape[-1], kind, width=args.width).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        target = torch.as_tensor(y[fit_indices], device=device)
        rng = np.random.default_rng(args.seed)
        best_ap, best_epoch = -1., 1
        curve = []
        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(len(fit_indices))
            total_loss = 0.
            for start in range(0, len(order), args.batch_size):
                idx = torch.as_tensor(order[start:start + args.batch_size], device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(*(a[idx] for a in fit_data))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total_loss += loss.item() * len(idx)
            item = {"epoch": epoch, "train_loss": total_loss / len(order)}
            if select:
                ap = float(average_precision_score(y[eval_indices], predict(model, eval_data)))
                item["inner_ap"] = ap
                if ap > best_ap:
                    best_ap, best_epoch = ap, epoch
            curve.append(item)
            if select and epoch - best_epoch >= args.patience:
                break
        return model, preprocessing, eval_data, best_epoch if select else epochs, curve

    for kind in args.models:
        for quarter in args.quarters:
            started = time.monotonic()
            fit_idx = np.flatnonzero((train.forecast_period < quarter).to_numpy())
            outer_idx = np.flatnonzero((train.forecast_period == quarter).to_numpy())
            inner_quarter = train.iloc[fit_idx].forecast_period.max()
            inner_fit = np.flatnonzero((train.forecast_period < inner_quarter).to_numpy())
            inner_eval = np.flatnonzero((train.forecast_period == inner_quarter).to_numpy())
            _, _, _, epochs, curve = fit(kind, inner_fit, inner_eval, args.max_epochs, True)
            model, preprocessing, outer_data, _, refit_curve = fit(kind, fit_idx, outer_idx, epochs, False)
            p = predict(model, outer_data)
            name = f"{kind}_{quarter}"
            checkpoint = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(checkpoint, args.out / f"{name}.pt")
            joblib.dump(preprocessing, args.out / f"{name}_preprocessing.joblib")
            reloaded = InspectionNet(outer_data[0].shape[1], preprocessing.cardinalities,
                                     events.shape[-1], kind, width=args.width).to(device)
            reloaded.load_state_dict(torch.load(args.out / f"{name}.pt", map_location=device, weights_only=True))
            restored = joblib.load(args.out / f"{name}_preprocessing.joblib")
            np.testing.assert_array_equal(p, predict(reloaded, prepare(outer_idx, restored)))
            record = {"model": kind, "seed": args.seed, "quarter": quarter,
                      "inner_quarter": inner_quarter, "selected_epochs": epochs,
                      "ap": float(average_precision_score(y[outer_idx], p)),
                      "train_n": len(fit_idx), "valid_n": len(outer_idx),
                      "parameters": sum(a.numel() for a in model.parameters()),
                      "seconds": time.monotonic() - started, "loaded_forward_exact": True}
            metrics.append(record)
            pred = train.iloc[outer_idx][["row_id", "site_id", "forecast_period", "target"]].copy()
            pred["model"], pred["prediction"] = kind, p
            predictions.append(pred)
            (args.out / f"{name}_curve.json").write_text(json.dumps({"selection": curve, "refit": refit_curve}, indent=2) + "\n")
            pd.DataFrame(metrics).to_csv(args.out / "fold_metrics.csv", index=False)
            pd.concat(predictions, ignore_index=True).to_csv(args.out / "predictions.csv", index=False)
            print(json.dumps(record), flush=True)
    summary = pd.DataFrame(metrics).groupby("model", as_index=False).agg(mean_ap=("ap", "mean"))
    summary.to_csv(args.out / "summary.csv", index=False)
    protocol["status"] = "complete"
    protocol["artifact_sha256"] = {p.name: sha(p) for p in sorted(args.out.iterdir()) if p.name != "protocol.json"}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
