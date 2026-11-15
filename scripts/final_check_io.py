"""Read frozen OOT forecasts and validate local final submissions."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from spend_baselines.suite import job_key, metric
from spend_baselines.suite_data import digest
from spend_baselines.cli import write_submission


class FrozenReference:
    def __init__(self, directory, expected, archive=None):
        self.directory = Path(directory)
        self.archive = zipfile.ZipFile(archive) if archive else None
        self.manifest = self.json("run_manifest.json")
        if self.manifest["definition"] != expected:
            raise ValueError("Frozen suite definition differs; do not overwrite the original run")
        self.plan = self.json("plan.json")

    def read(self, name):
        return self.archive.read(name) if self.archive else (self.directory / name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def prediction(self, spec, objects):
        prefix = f"jobs/{job_key(spec)}"
        result = self.json(prefix + "/result.json")
        if result["job"] != spec:
            raise ValueError("Frozen result specification mismatch")
        payload = self.read(prefix + "/predictions.npz")
        import hashlib
        if hashlib.sha256(payload).hexdigest() != result["sha256"]["predictions.npz"]:
            raise ValueError("Corrupt frozen predictions")
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            ids, z = saved["Id"].astype(str), saved["z"].astype(float)
        expected = objects.loc[objects.official & objects.as_of_date.eq(pd.Timestamp(spec["date"]))].set_index("Id")
        if len(ids) != len(set(ids)) or set(ids) != set(expected.index):
            raise ValueError("Frozen validation population mismatch")
        order = pd.Index(ids).get_indexer(expected.index)
        z = z[order]
        actual = metric(expected.spend_30d.to_numpy(float), z)
        if abs(actual["RMSLE"] - result["RMSLE"]) > 1e-9:
            raise ValueError("Frozen score does not reproduce")
        return expected.index.to_numpy(str), z

    def mean_prediction(self, name, date, objects, dataset="D2", representation="F2"):
        family, task, profile = name.split(":")
        jobs = [j for j in self.plan if j["date"] == date and j.get("dataset") == dataset
                and j.get("representation") == representation and j["family"] == family
                and j["task"] == task and j.get("profile", "fixed") == profile
                and j["kind"] in {"tabular", "neural"}]
        wanted = self.manifest["definition"]["config"]["seeds"] if profile != "fixed" else [42]
        if sorted(j["seed"] for j in jobs) != sorted(wanted):
            raise ValueError(f"Incomplete frozen seed definition: {name}/{date}")
        predictions = [self.prediction(j, objects) for j in jobs]
        ids = predictions[0][0]
        if any(not np.array_equal(ids, p[0]) for p in predictions):
            raise ValueError("Frozen seed order mismatch")
        return ids, np.mean([p[1] for p in predictions], axis=0)


def validate_submission(path, sample):
    frame = pd.read_csv(path, dtype={"Id": "string"})
    if frame.columns.tolist() != ["Id", "spend_30d"]:
        raise ValueError("Wrong submission columns")
    if not frame.Id.is_unique or frame.Id.tolist() != sample.Id.tolist():
        raise ValueError("Submission IDs/order differ from sample")
    if not np.isfinite(frame.spend_30d).all() or frame.spend_30d.lt(0).any():
        raise ValueError("Invalid submission values")
    return frame


def locked_submission(reference_dir, name, data_dir):
    sample = pd.read_csv(Path(data_dir) / "sample_submission.csv", dtype={"Id": "string"})
    test = pd.read_csv(Path(data_dir) / "test.csv", parse_dates=["as_of_date"])
    date = str(test.as_of_date.min().date())
    key = job_key({"candidate": name, "dataset": "D2", "representation": "F2", "date": date})
    directory = Path(reference_dir) / "candidates" / key
    result = json.loads((directory / "result.json").read_text())
    if (result["candidate"], result["dataset"], result["representation"]) != (name, "D2", "F2"):
        raise ValueError("Wrong locked final candidate")
    for filename, sha in result["sha256"].items():
        if digest(directory / filename) != sha:
            raise ValueError("Corrupt locked final candidate")
    return validate_submission(directory / "submission.csv", sample)


def export_locked_submissions(reference_dir, config, data_dir, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(Path(data_dir) / "sample_submission.csv", dtype={"Id": "string"})
    main = locked_submission(reference_dir, config["locked_primary"], data_dir)
    main.to_csv(destination / "joint_mean3.csv", index=False)
    if config["refit_reserve"]:
        wide = locked_submission(reference_dir, config["locked_reserve"], data_dir)
        wide.to_csv(destination / "wide_mean3.csv", index=False)
        for weight in config["blend_joint_weights"]:
            if weight == 1:
                continue  # The primary file already contains this exact candidate.
            z = weight * np.log1p(main.spend_30d.to_numpy()) + (1 - weight) * np.log1p(wide.spend_30d.to_numpy())
            filename = f"joint{round(weight*100)}_wide{round((1-weight)*100)}.csv"
            write_submission(main[["Id"]], z, destination / filename, sample.Id)
            validate_submission(destination / filename, sample)
    return destination / "joint_mean3.csv"
