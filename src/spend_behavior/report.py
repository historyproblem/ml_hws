"""Read verified frozen predictions; paired, customer-cluster comparisons."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from spend_baselines.suite_data import BASE_COLUMNS, digest


class Reference:
    def __init__(self, path, data_dir):
        self.path = Path(path)
        self.archive = zipfile.ZipFile(path) if self.path.is_file() else None
        manifest = json.loads(self.read("run_manifest.json"))
        definition = manifest["definition"]
        for name, sha in definition["input_sha256"].items():
            if digest(Path(data_dir) / name) != sha:
                raise ValueError(f"Reference inputs differ: {name}")
        # Check the source of the frozen baseline, not the new experiment package.
        for name, sha in definition["source_sha256"].items():
            source = Path(__file__).parents[1] / "spend_baselines" / name
            if digest(source) != sha:
                raise ValueError(f"Frozen baseline code differs: {name}")
        self.members = {}
        names = self.archive.namelist() if self.archive else [str(p.relative_to(self.path)) for p in (self.path / "jobs").glob("*/result.json")]
        for name in names:
            if not name.startswith("jobs/") or not name.endswith("/result.json"):
                continue
            r = json.loads(self.read(name))
            j = r["job"]
            if (j.get("kind") == "neural" and j.get("dataset") == "D2" and
                j.get("representation") == "F2" and j.get("family") == "mlp" and
                j.get("task") == "hurdle_joint" and j.get("profile") == "compact_l2" and
                j.get("seed") in (42, 123, 2026) and not j.get("test")):
                self.members.setdefault(j["date"], []).append((name.rsplit("/", 1)[0], r))

    def read(self, name):
        return self.archive.read(name) if self.archive else (self.path / name).read_bytes()

    def predictions(self, date, objects):
        members = self.members.get(str(pd.Timestamp(date).date()), [])
        if sorted(r["job"]["seed"] for _, r in members) != [42, 123, 2026]:
            raise ValueError(f"Reference requires all three Joint seeds at {date}")
        zs, mus = [], []
        for directory, result in sorted(members, key=lambda x: x[1]["job"]["seed"]):
            content = self.read(directory + "/predictions.npz")
            if hashlib.sha256(content).hexdigest() != result["sha256"]["predictions.npz"]:
                raise ValueError("Corrupt reference predictions")
            with np.load(io.BytesIO(content), allow_pickle=False) as p:
                frame = pd.DataFrame({"Id": p["Id"], "customer_id": p["customer_id"],
                                      "z": p["z"], "p": p["probability"]}).set_index("Id")
            if not frame.index.is_unique or set(frame.index) != set(objects.Id):
                raise ValueError("Reference validation population differs")
            frame = frame.loc[objects.Id]
            if not np.array_equal(frame.customer_id.to_numpy(str), objects.customer_id.to_numpy(str)):
                raise ValueError("Reference customer keys differ")
            if not np.isfinite(frame[["z", "p"]]).all().all() or (frame.z < 0).any() or not frame.p.between(1e-10, 1).all():
                raise ValueError("Invalid reference Joint heads")
            score = np.sqrt(np.mean((frame.z.to_numpy() - np.log1p(objects.spend_30d.to_numpy())) ** 2))
            if abs(score - result["RMSLE"]) > 1e-9:
                raise ValueError("Reference RMSLE does not reproduce")
            zs.append(frame.z.to_numpy())
            mus.append(frame.z.to_numpy() / frame.p.to_numpy())
        return np.mean(zs, axis=0), np.mean(mus, axis=0)


def cluster_interval(frame, repeats=2000, seed=2026):
    errors = pd.DataFrame({"customer": frame.customer_id,
                           "e": (frame.z - frame.target_z) ** 2,
                           "b": (frame.reference_z - frame.target_z) ** 2})
    grouped = errors.groupby("customer").agg(e=("e", "sum"), b=("b", "sum"), n=("e", "size"))
    a = grouped.to_numpy(float)
    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(repeats):
        sample = a[rng.integers(0, len(a), len(a))].sum(0)
        delta.append(np.sqrt(sample[0] / sample[2]) - np.sqrt(sample[1] / sample[2]))
    return np.quantile(delta, [0.025, 0.975]).tolist()


def comparison(frame, repeats):
    e = (frame.z - frame.target_z) ** 2
    b = (frame.reference_z - frame.target_z) ** 2
    lo, hi = cluster_interval(frame, repeats)
    return {"n": len(frame), "RMSLE": float(np.sqrt(e.mean())),
            "reference_RMSLE": float(np.sqrt(b.mean())),
            "delta": float(np.sqrt(e.mean()) - np.sqrt(b.mean())),
            "delta_ci_low": lo, "delta_ci_high": hi}


def write_report(directory, data, config):
    directory = Path(directory)
    paths = sorted((directory / "predictions").glob("*/*.csv.gz"))
    frames = [pd.read_csv(p, dtype={"Id": str, "customer_id": str}) for p in paths]
    if not frames:
        return
    all_rows = pd.concat(frames, ignore_index=True)
    if all_rows.duplicated(["candidate", "Id"]).any():
        raise ValueError("Duplicate predictions in report")
    scores, slices = [], []
    counts = pd.Series(np.rint(np.expm1(data.base.features[:, BASE_COLUMNS.index("n_180")])).astype(int), index=data.objects.Id)
    for name, rows in all_rows.groupby("candidate", sort=True):
        groups = [(date, g) for date, g in rows.groupby("date")]
        late = rows[rows.date.isin(["2018-09-01", "2018-10-01"])]
        if set(late.date) == {"2018-09-01", "2018-10-01"}:
            groups.append(("pooled_SepOct", late))
        if set(config["validation_dates"]) <= set(rows.date):
            groups.append((f"pooled_{len(config['validation_dates'])}_dates", rows))
        for date, g in groups:
            scores.append({"candidate": name, "date": date, **comparison(g, config["bootstrap_repeats"])})
            n = counts.loc[g.Id].to_numpy()
            for label, mask in (("days_0", n == 0), ("days_1_2", (n >= 1) & (n <= 2)),
                                ("days_3_plus", n >= 3), ("zero", g.target_z.to_numpy() == 0),
                                ("positive", g.target_z.to_numpy() > 0)):
                slices.append({"candidate": name, "date": date, "slice": label, "n": int(mask.sum()),
                               "RMSLE": float(np.sqrt(np.mean((g.z.to_numpy()[mask] - g.target_z.to_numpy()[mask]) ** 2))) if mask.any() else None})
    table = pd.DataFrame(scores)
    table.to_csv(directory / "scores.csv", index=False)
    pd.DataFrame(slices).to_csv(directory / "error_slices.csv", index=False)
    # Shared work (OOF/preparation) is recorded separately, never charged as zero.
    timing = []
    for p in sorted((directory / "results").glob("*/*.json")):
        r = json.loads(p.read_text())
        timing.append({k: r[k] for k in ("candidate", "date", "train_seconds", "inference_seconds")})
    pd.DataFrame(timing).to_csv(directory / "timing.csv", index=False)
    transfer = []
    for date, rows in data.objects.loc[data.objects.official | data.objects.is_test].groupby("as_of_date"):
        from .data import COLUMNS
        ix = rows.index.to_numpy()
        transfer.append({"date": str(date.date()), "n": len(rows), "test": bool(rows.is_test.all()),
                         "prior_year_coverage": float(np.mean(data.extra[ix, COLUMNS.index("year_coverage")])),
                         "global_log_season": float(np.mean(data.extra[ix, COLUMNS.index("global_season")]))})
    pd.DataFrame(transfer).to_csv(directory / "season_transfer.csv", index=False)
    late = table[table.date.eq("pooled_SepOct")].sort_values("RMSLE")
    lines = ["# Customer behaviour experiments", "", "Frozen control: D2/F2 Joint mean3 (42/123/2026).",
             "Delta = candidate minus Joint; negative is better. Customer-cluster bootstrap is conditional on these reused dates; it does not correct multiple comparisons, model selection or temporal drift.", "",
             "| Candidate | Sep–Oct RMSLE | Delta | 95% CI |", "|---|---:|---:|---| "]
    for r in late.itertuples():
        lines.append(f"| {r.candidate} | {r.RMSLE:.6f} | {r.delta:+.6f} | [{r.delta_ci_low:+.6f}, {r.delta_ci_high:+.6f}] |")
    lines += ["", "November test improvement is unobserved. June–October checks do not establish transfer to the November peak. No test labels or leaderboard tuning are used.",
              "", "A 0.001–0.003 gain whose interval includes zero remains inconclusive. Missing dates are not counted as a complete pooled result. The 0-day history slice is empty when eligibility requires a purchase in 180 days; see error_slices.csv.",
              "", "Per-date scores, zero/positive and history slices, paired row predictions, selection logs and separate fit/inference timings accompany this report."]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
