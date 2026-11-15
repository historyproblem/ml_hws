"""Reuse verified F2/history caches and current-best CUDA predictions read-only."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from spend_baselines.data import read_objects
from spend_baselines.suite_data import BASE_COLUMNS, FEATURES, CachedData, digest
from spend_behavior.data import HISTORY


def causal_history_check(objects, prior, half_life):
    """Independent vectorised audit of the four cached target-history features.

    No model fits and no OOF residual calculations. target_end == row time is
    available; overlapping synthetic snapshots never become history labels.
    """
    labels = objects.loc[objects.official].sort_values("target_end")
    origin = labels.target_end.min()
    times = (objects.as_of_date - origin).dt.total_seconds().to_numpy() / 86400
    ends = (labels.target_end - origin).dt.total_seconds().to_numpy() / 86400
    z = np.log1p(labels.spend_30d.to_numpy(float))
    n_global = np.searchsorted(ends, times, side="right")
    mean = np.r_[0., np.cumsum(z)][n_global] / np.maximum(n_global, 1)
    zero = np.r_[0., np.cumsum(z == 0)][n_global] / np.maximum(n_global, 1)
    result = np.empty((len(objects), 4), np.float32)
    groups = {str(cid): g for cid, g in labels.groupby("customer_id", sort=False)}
    for cid, rows in objects.groupby("customer_id", sort=False):
        ix = rows.index.to_numpy()
        past = groups.get(str(cid), labels.iloc[:0])
        dates = (past.target_end - origin).dt.total_seconds().to_numpy() / 86400
        values = np.log1p(past.spend_30d.to_numpy(float))
        n = np.searchsorted(dates, times[ix], side="right")
        scale = np.exp2(dates / half_life)
        decay = np.exp2(-times[ix] / half_life)
        weighted_sum = np.r_[0., np.cumsum(values * scale)][n] * decay
        weighted_n = np.r_[0., np.cumsum(scale)][n] * decay
        result[ix] = np.column_stack([
            np.log1p(n),
            (np.r_[0., np.cumsum(values)][n] + prior * mean[ix]) / (n + prior),
            (np.r_[0., np.cumsum(values == 0)][n] + prior * zero[ix]) / (n + prior),
            (weighted_sum + prior * mean[ix]) / (weighted_n + prior),
        ])
    return result


class FrozenRun:
    def __init__(self, path, data_dir, base_settings):
        self.path = Path(path)
        self.tar = tarfile.open(path, "r:gz") if self.path.is_file() else None
        self.members = {}
        if self.tar:
            for member in self.tar.getmembers():
                name = member.name.removeprefix("./")
                if member.isdir():
                    continue
                if not member.isfile() or Path(name).is_absolute() or ".." in Path(name).parts or name in self.members:
                    raise ValueError("Unsafe/duplicate reference archive entry")
                self.members[name] = member
        self.manifest_bytes = self.read("manifest.json")
        self.manifest = json.loads(self.manifest_bytes)
        cfg = self.manifest["config"]
        if self.manifest["device"] != "cuda" or cfg["seeds"] != [42, 123, 2026]:
            raise ValueError("Reference must contain the verified CUDA mean3")
        if cfg["joint_settings"] != base_settings or cfg["dataset"] != "D2":
            raise ValueError("Reference baseline settings differ")
        for name, sha in self.manifest["inputs"].items():
            if digest(Path(data_dir) / name) != sha:
                raise ValueError(f"Reference input differs: {name}")
        source_root = Path(__file__).parents[1]
        for name, sha in self.manifest["source_sha256"].items():
            if not name.startswith(("spend_baselines/", "spend_behavior/")) or ".." in Path(name).parts:
                raise ValueError("Unexpected reference source path")
            if digest(source_root / name) != sha:
                raise ValueError(f"Frozen implementation differs: {name}")
        self.train = read_objects(Path(data_dir) / "train.csv")
        self.frames = {}
        self.prediction_hashes = {}
        # Record metadata hashes without opening late predictions for selection.
        for date in cfg["validation_dates"]:
            result = json.loads(self.read(f"results/joint_history_mean3/{date}.json"))
            self.prediction_hashes[date] = result["predictions_sha256"]
        self.history_marker = json.loads(self.read("histories.json"))
        for fold in self.history_marker["folds"]:
            if pd.Timestamp(fold["max_target_end"]) > pd.Timestamp(fold["date"]):
                raise ValueError("Reference OOF fold crosses its cutoff")

    def read(self, name):
        return (self.tar.extractfile(self.members[name]).read() if self.tar else
                (self.path / name).read_bytes())

    def frame(self, date):
        date = str(pd.Timestamp(date).date())
        if date not in self.frames:
            payload = self.read(f"predictions/joint_history_mean3/{date}.csv.gz")
            if hashlib.sha256(payload).hexdigest() != self.prediction_hashes[date]:
                raise ValueError("Corrupt reference predictions")
            frame = pd.read_csv(io.BytesIO(payload), compression="gzip",
                                dtype={"Id": str, "customer_id": str})
            if not frame.date.eq(date).all() or not frame.candidate.eq("joint_history_mean3").all():
                raise ValueError("Reference candidate/date differs")
            truth = self.train.loc[self.train.as_of_date.eq(date)].set_index("Id")
            if not frame.Id.is_unique or set(frame.Id) != set(truth.index):
                raise ValueError("Reference validation population differs")
            truth = truth.loc[frame.Id]
            if not np.array_equal(frame.customer_id.to_numpy(), truth.customer_id.to_numpy()):
                raise ValueError("Reference customer keys differ")
            np.testing.assert_allclose(frame.target_z, np.log1p(truth.spend_30d.to_numpy()), rtol=0, atol=2e-14)
            if not np.isfinite(frame.z).all() or (frame.z < 0).any():
                raise ValueError("Invalid reference forecasts")
            frame["reference_z"] = frame.z
            frame["candidate"] = "baseline"
            frame["delta_squared_log_error"] = 0.
            self.frames[date] = frame.set_index("Id", drop=False)
        return self.frames[date]


class TuningData:
    def __init__(self, reference, explicit_cache=None, smoke=False):
        candidates = ([Path(explicit_cache)] if explicit_cache else
                      [reference.path / "base_cache", Path(reference.manifest["base_cache"] or ".")])
        self.directory = next((p for p in candidates if (p / "ready.json").is_file()), None)
        if self.directory is None:
            raise FileNotFoundError("No F2 cache found; pass --base-cache outputs/v100-focused/cache")
        ready = json.loads((self.directory / "ready.json").read_text())
        ident = ready["identity"]
        if ident["input_sha256"] != reference.manifest["inputs"] or ready["columns"] != BASE_COLUMNS:
            raise ValueError("Cached F2 inputs/columns differ")
        if "D2" not in ident["config"]["datasets"] or ident["config"].get("smoke_customers"):
            raise ValueError("A complete D2 cache is required")
        for name, sha in ready["sha256"].items():
            if digest(self.directory / name) != sha:
                raise ValueError(f"Corrupt F2 cache: {name}")
        self.base = CachedData(self.directory)
        self.objects = self.base.objects
        official = self.objects.loc[self.objects.official].set_index("Id")
        known = reference.train.set_index("Id")
        if not official.index.is_unique or set(official.index) != set(known.index):
            raise ValueError("Cached official population differs")
        known = known.loc[official.index]
        for name in ("customer_id", "as_of_date", "target_end"):
            if not np.array_equal(official[name].to_numpy(), known[name].to_numpy()):
                raise ValueError(f"Cached keys differ: {name}")
        np.testing.assert_allclose(official.spend_30d, known.spend_30d, rtol=0, atol=.011)
        payload = reference.read("histories.npz")
        if hashlib.sha256(payload).hexdigest() != reference.history_marker["sha256"]:
            raise ValueError("Corrupt history cache")
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            self.history = data["history"]
        if self.history.shape != (len(self.objects), len(HISTORY)):
            raise ValueError("History population/columns differ")
        cfg = reference.manifest["config"]
        audited = causal_history_check(self.objects, cfg["history_prior"], cfg["history_half_life"])
        np.testing.assert_allclose(self.history, audited, rtol=0, atol=1e-6)
        self.active = np.ones(len(self.objects), dtype=bool)
        if smoke:
            customers = sorted(self.objects.customer_id.unique(), key=lambda c: hashlib.sha256(c.encode()).hexdigest())[:64]
            self.active = self.objects.customer_id.isin(customers).to_numpy()
        self.identity = {"ready_sha256": digest(self.directory / "ready.json"),
                         "history_sha256": reference.history_marker["sha256"],
                         "columns": FEATURES["F2"] + HISTORY,
                         "rows": len(self.objects), "active_rows": int(self.active.sum())}

    def matrix(self, indices):
        return np.column_stack([self.base.matrix("F2", indices)[0], self.history[indices]])

    def valid(self, date, testing=False):
        return np.flatnonzero(self.active & self.objects.as_of_date.eq(pd.Timestamp(date)).to_numpy() &
                              (self.objects.is_test if testing else self.objects.official).to_numpy())
