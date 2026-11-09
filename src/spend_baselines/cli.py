"""Commands for full temporal validation and generating a local submission."""
import argparse
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from .data import load_dataset, feature_matrix
from .models import fit_model, predict_logs, rmsle_from_logs

MODEL_NAMES = ["zero", "logmean", "ridge", "forest", "mlp"]


def temporal_masks(objects, date):
    date = pd.Timestamp(date)
    fit = objects.target_end.le(date)
    valid = objects.as_of_date.eq(date)
    if (fit & valid).any():
        raise ValueError("Overlapping temporal train and validation splits")
    if not fit.any() or not valid.any():
        raise ValueError(f"Empty temporal split for {date.date()}")
    return fit, valid


def write_submission(test, log_predictions, path, sample_ids=None):
    predictions = np.expm1(np.asarray(log_predictions))
    if len(predictions) != len(test) or not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Submission must have one finite nonnegative prediction per test row")
    if not test.Id.is_unique:
        raise ValueError("Duplicate test Id")
    out = pd.DataFrame({"Id": test.Id.to_numpy(), "spend_30d": predictions})
    if sample_ids is not None:
        if not sample_ids.is_unique or set(sample_ids) != set(out.Id):
            raise ValueError("Sample submission and test identifiers differ")
        out = out.set_index("Id").loc[sample_ids].reset_index()
    out.to_csv(path, index=False)
    return out


def manifest(args, config, stats):
    sources = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = {}
    for name in ["transactions.csv", "train.csv", "test.csv", "sample_submission.csv"]:
        p = args.data_dir / name
        if p.is_file():
            inputs[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    libraries = {}
    for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl", "torch"]:
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            pass
    models = args.models if args.command == "validate" else (["mlp"] if args.command == "mlp-experiment" else [args.model])
    return {"command": args.command, "models": models,
            "config": config, "device": args.device,
            "threads": args.threads, "python": platform.python_version(),
            "platform": platform.platform(), "libraries": libraries,
            "source_sha256": sources, "input_sha256": inputs, "data_checks": stats}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "submit", "mlp-experiment"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/baselines.json"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["logmean", "ridge", "forest", "mlp"])
    parser.add_argument("--model", choices=MODEL_NAMES, default="ridge", help="Single model for submit")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--features", choices=["rfm", "extended"])
    parser.add_argument("--epochs", type=int, help="Explicit manual override for MLP training epochs")
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/mlp_experiments.json"))
    parser.add_argument("--variants", nargs="+", help="Explicit named MLP experiments")
    parser.add_argument("--seeds", nargs="+", type=int, help="Explicit repeated-run seeds")
    parser.add_argument("--resume", action="store_true", help="Resume an identical interrupted MLP experiment run")
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = Path("outputs/mlp-experiments" if args.command == "mlp-experiment" else "outputs/baselines")
    if args.threads < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error("threads and epochs must be positive")
    if args.command != "mlp-experiment" and (args.variants or args.seeds or args.resume):
        parser.error("--variants, --seeds and --resume are only for mlp-experiment")
    config = json.loads(args.config.read_text())
    if args.features:
        config["feature_group"] = args.features
    if args.epochs:
        config["mlp"]["epochs"] = args.epochs
    if args.command == "mlp-experiment":
        from .experiments import run_mlp_experiments
        return run_mlp_experiments(args, config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading official data and checking all known training targets", flush=True)
    train, test, history, stats = load_dataset(args.data_dir)
    print(json.dumps(stats), flush=True)
    info = manifest(args, config, stats)
    (args.out_dir / "run_manifest.json").write_text(json.dumps(info, indent=2))
    print("Building features with per-row historical cutoffs", flush=True)
    features = history.build_features(train)
    x = feature_matrix(features, config["feature_group"])

    if args.command == "validate":
        scores, predictions = [], []
        for date in config["validation_dates"]:
            fit, valid = temporal_masks(train, date)
            target = train.loc[valid, "spend_30d"].to_numpy()
            prediction_rows = train.loc[valid, ["Id", "customer_id", "as_of_date", "spend_30d"]].copy()
            for name in args.models:
                started = time.monotonic()
                fitted = None if name in {"zero", "logmean"} else fit_model(
                    name, x[fit], train.loc[fit, "spend_30d"].to_numpy(), config,
                    device=args.device, threads=args.threads,
                )
                z = predict_logs(name, fitted, x[valid], features.loc[valid])
                score = rmsle_from_logs(target, z)
                elapsed = time.monotonic() - started
                prediction_rows[name] = np.expm1(z)
                scores.append({"date": date, "model": name, "features": config["feature_group"],
                               "n_train": int(fit.sum()), "n_valid": int(valid.sum()),
                               "RMSLE": score, "seconds": elapsed})
                if name == "mlp":
                    (args.out_dir / f"mlp_train_{date}.json").write_text(json.dumps(fitted.history, indent=2))
                pd.DataFrame(scores).to_csv(args.out_dir / "scores.csv", index=False)
                print(f"{date} {name}: RMSLE={score:.6f} ({elapsed:.1f}s)", flush=True)
            predictions.append(prediction_rows)
            pd.concat(predictions, ignore_index=True).to_csv(args.out_dir / "validation_predictions.csv", index=False)
        print(pd.DataFrame(scores).pivot(index="model", columns="date", values="RMSLE").to_string(), flush=True)
    else:
        cutoff = test.as_of_date.min()
        fit = train.target_end.le(cutoff)
        if not fit.any():
            raise ValueError("No training labels available before test prediction date")
        test_features = history.build_features(test)
        xt = feature_matrix(test_features, config["feature_group"])
        name = args.model
        fitted = None if name in {"zero", "logmean"} else fit_model(
            name, x[fit], train.loc[fit, "spend_30d"].to_numpy(), config,
            device=args.device, threads=args.threads,
        )
        z = predict_logs(name, fitted, xt, test_features)
        sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"Id": "string"})
        dest = args.out_dir / f"submission_{name}.csv"
        out = write_submission(test, z, dest, sample_ids=sample.Id)
        print(f"Wrote {len(out)} predictions to {dest}; no Kaggle upload was performed", flush=True)


if __name__ == "__main__":
    main()
