"""Weighted tabular and neural models; all regressions return log1p units."""

import os
import random
from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler


class Preprocessor:
    def fit(self, x, w):
        self.imputer = SimpleImputer(
            strategy="median", add_indicator=False, keep_empty_features=True
        )
        a = self.imputer.fit_transform(x)
        self.missing = np.flatnonzero(np.isnan(x).any(axis=0))
        a = np.c_[a, np.isnan(x[:, self.missing])]
        self.scaler = StandardScaler().fit(a, sample_weight=w)
        return self

    def transform(self, x):
        # One arithmetic path for fitting-time and portable-checkpoint inference.
        return self.apply_state(x, self.state())

    def state(self):
        return {
            "fill": self.imputer.statistics_.tolist(),
            "missing": self.missing.tolist(),
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
        }

    @staticmethod
    def apply_state(x, state):
        a = np.where(np.isnan(x), np.asarray(state["fill"]), x)
        a = np.c_[a, np.isnan(x[:, state["missing"]])]
        return ((a - np.asarray(state["mean"])) / np.asarray(state["scale"])).astype(
            np.float32
        )


class Tabular:
    def __init__(self, family, task, settings, seed=42, threads=2):
        self.family, self.task, self.settings, self.seed, self.threads = (
            family,
            task,
            settings,
            seed,
            threads,
        )

    def fit(self, x, y, w, **unused):
        selected = y > 0 if self.task == "positive" else np.ones(len(y), dtype=bool)
        selected &= w > 0
        x, y, w = x[selected], y[selected], w[selected]
        if len(y) == 0:
            raise ValueError("No observations for fitted task")
        self.preprocessor = Preprocessor().fit(x, w)
        a = self.preprocessor.transform(x)
        target = (y > 0).astype(float) if self.task == "probability" else np.log1p(y)
        self.constant = None
        if len(np.unique(target)) == 1:
            self.constant = float(target[0])
            return self
        self.spline = None
        if self.family == "spline":
            self.spline = SplineTransformer(
                n_knots=self.settings["spline_knots"],
                degree=3,
                include_bias=False,
                extrapolation="constant",
            )
            a = self.spline.fit_transform(a)
        if self.family in {"linear", "spline"}:
            self.model = (
                LogisticRegression(
                    C=self.settings["logistic_C"], max_iter=2000, random_state=self.seed
                )
                if self.task == "probability"
                else Ridge(alpha=self.settings["ridge_alpha"])
            )
        elif self.family == "forest":
            cls = (
                RandomForestClassifier
                if self.task == "probability"
                else RandomForestRegressor
            )
            self.model = cls(
                **self.settings["forest"], random_state=self.seed, n_jobs=self.threads
            )
        else:
            raise ValueError(self.family)
        self.model.fit(a, target, sample_weight=w)
        return self

    def predict(self, x):
        if self.constant is not None:
            return np.full(len(x), self.constant)
        a = self.preprocessor.transform(x)
        if self.spline is not None:
            a = self.spline.transform(a)
        result = (
            self.model.predict_proba(a)[:, 1]
            if self.task == "probability"
            else self.model.predict(a)
        )
        return (
            np.clip(result, 0, 1)
            if self.task == "probability"
            else np.maximum(result, 0)
        )


def make_network(n_features, architecture, task, settings, sequence_shape=None):
    import torch

    nn = torch.nn
    widths = settings["hidden_sizes"]
    out = 3 if task == "ziln" else (2 if task.startswith("hurdle") else 1)
    dropout = settings.get("dropout", 0.0)

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.sequence_shape = sequence_shape
            self.architecture = architecture
            self.wide = nn.Linear(n_features, out) if architecture == "wide" else None
            if architecture in {"gru", "lstm", "tcn"}:
                if sequence_shape is None:
                    raise ValueError(
                        "A sequence encoder needs a sequence representation"
                    )
                self.tab_size = n_features - sequence_shape[0] * sequence_shape[1]
                h = widths[0]
                if architecture == "gru":
                    self.encoder = nn.GRU(sequence_shape[1], h, batch_first=True)
                elif architecture == "lstm":
                    self.encoder = nn.LSTM(sequence_shape[1], h, batch_first=True)
                else:
                    self.encoder = nn.ModuleList(
                        [
                            nn.Conv1d(
                                sequence_shape[1] if i == 0 else h,
                                h,
                                3,
                                padding=2 * (2**i),
                                dilation=2**i,
                            )
                            for i in range(3)
                        ]
                    )
                previous = h + self.tab_size
                layers = []
                for width in widths[1:]:
                    layers.extend(
                        [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
                    )
                    previous = width
                self.body = nn.Sequential(*layers)
            elif architecture == "residual":
                self.layers = nn.ModuleList()
                self.skips = nn.ModuleList()
                previous = n_features
                for width in widths:
                    self.layers.append(
                        nn.Sequential(
                            nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)
                        )
                    )
                    self.skips.append(
                        nn.Linear(previous, width, bias=False)
                        if previous != width
                        else nn.Identity()
                    )
                    previous = width
            else:
                layers = []
                previous = n_features
                for width in widths:
                    layers.extend(
                        [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
                    )
                    previous = width
                self.body = nn.Sequential(*layers)
            self.output = nn.Linear(previous, out)

        def forward(self, x):
            if self.architecture in {"gru", "lstm", "tcn"}:
                tab = x[:, : self.tab_size]
                seq = x[:, self.tab_size :].reshape(-1, *self.sequence_shape)
                if self.architecture == "tcn":
                    h = seq.transpose(1, 2)
                    for layer in self.encoder:
                        h = torch.relu(layer(h)[..., : seq.shape[1]])
                    h = h.mean(dim=-1)
                else:
                    h = self.encoder(seq)[0][:, -1, :]
                h = self.body(torch.cat([tab, h], dim=1))
            elif self.architecture == "residual":
                h = x
                for layer, skip in zip(self.layers, self.skips):
                    h = (layer(h) + skip(h)) / 2**0.5
            else:
                h = self.body(x)
            raw = self.output(h)
            return raw + self.wide(x) if self.wide is not None else raw

    return Network()


def output_prediction(raw, task):
    import torch

    if task == "probability":
        return torch.sigmoid(raw[:, 0])
    if task.startswith("hurdle"):
        return torch.sigmoid(raw[:, 0]) * torch.nn.functional.softplus(raw[:, 1])
    if task == "ziln":
        nodes, weights = np.polynomial.hermite.hermgauss(32)
        mu = raw[:, 1, None]
        sigma = torch.nn.functional.softplus(raw[:, 2, None]) + 0.05
        values = torch.nn.functional.softplus(
            mu
            + sigma
            * torch.as_tensor(nodes * 2**0.5, device=raw.device, dtype=raw.dtype)
        )
        expectation = values @ torch.as_tensor(
            weights / np.sqrt(np.pi), device=raw.device, dtype=raw.dtype
        )
        return torch.sigmoid(raw[:, 0]) * expectation
    return raw[:, 0].clamp_min(0)


def neural_loss(raw, y, w, task):
    import torch

    functional = torch.nn.functional
    z = torch.log1p(y)
    positive = y > 0

    def average(loss, weights=w):
        return (loss * weights).sum() / weights.sum().clamp_min(1e-12)

    if task == "probability":
        return average(
            functional.binary_cross_entropy_with_logits(
                raw[:, 0], positive.float(), reduction="none"
            )
        )
    if task.startswith("hurdle") or task == "ziln":
        loss = average(
            functional.binary_cross_entropy_with_logits(
                raw[:, 0], positive.float(), reduction="none"
            )
        )
        if positive.any():
            if task == "ziln":
                sigma = functional.softplus(raw[positive, 2]) + 0.05
                logy = y[positive].log()
                cost = (
                    sigma.log()
                    + logy
                    + 0.5 * ((logy - raw[positive, 1]) / sigma) ** 2
                    + 0.5 * np.log(2 * np.pi)
                )
            else:
                cost = (functional.softplus(raw[positive, 1]) - z[positive]) ** 2
            loss = loss + average(cost, w[positive])
        if task == "hurdle_joint":
            loss = loss + average((output_prediction(raw, task) - z) ** 2)
        return loss
    # Direct regression deliberately has no clamp in its training loss.
    return average((raw[:, 0] - z) ** 2)


class Neural:
    def __init__(
        self,
        architecture,
        task,
        settings,
        seed=42,
        device="cpu",
        threads=2,
        sequence_shape=None,
    ):
        self.architecture, self.task, self.settings = (
            architecture,
            task,
            deepcopy(settings),
        )
        self.seed, self.device, self.threads, self.sequence_shape = (
            seed,
            device,
            threads,
            sequence_shape,
        )

    def fit(self, x, y, w, validation=None, epochs=None):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        keep = (w > 0) & ((y > 0) if self.task == "positive" else True)
        x, y, w = x[keep], y[keep], w[keep]
        if not len(y):
            raise ValueError("Empty neural training task")
        # Sequence models need the sequence at the end: do not append missing flags.
        self.preprocessor = Preprocessor().fit(x, w)
        if self.sequence_shape is not None:
            self.preprocessor.missing = np.array([], dtype=int)
            a = self.preprocessor.imputer.transform(x)
            self.preprocessor.scaler = StandardScaler().fit(a, sample_weight=w)
        a = self.preprocessor.transform(x)
        self.network = make_network(
            a.shape[1], self.architecture, self.task, self.settings, self.sequence_shape
        ).to(self.device)
        tx = torch.from_numpy(a).to(self.device)
        ty = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        tw = torch.as_tensor(w, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.settings["learning_rate"],
            weight_decay=self.settings["weight_decay"],
        )
        if validation is not None:
            vx, vy = validation
            if self.task == "positive":
                select = vy > 0
                vx, vy = vx[select], vy[select]
            if not len(vy):
                raise ValueError("Empty neural inner validation task")
            va = torch.from_numpy(self.preprocessor.transform(vx)).to(self.device)
        self.history = []
        self.best_epoch = 0
        best = np.inf
        state = None
        stale = 0
        for epoch in range(epochs or self.settings["max_epochs"]):
            self.network.train()
            total = 0.0
            mass = 0.0
            for indices in torch.randperm(len(tx), device=self.device).split(
                self.settings["batch_size"]
            ):
                optimizer.zero_grad(set_to_none=True)
                loss = neural_loss(
                    self.network(tx[indices]), ty[indices], tw[indices], self.task
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite neural training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
                optimizer.step()
                total += float(loss.detach()) * len(indices)
                mass += len(indices)
            record = {"epoch": epoch + 1, "train_loss": total / mass}
            if validation is not None:
                predicted = self._predict_tensor(va)
                if self.task == "probability":
                    prob = np.clip(predicted, 1e-8, 1 - 1e-8)
                    binary = (vy > 0).astype(float)
                    score = float(
                        -np.mean(binary * np.log(prob) + (1 - binary) * np.log1p(-prob))
                    )
                else:
                    score = float(np.sqrt(np.mean((np.log1p(vy) - predicted) ** 2)))
                if not np.isfinite(score):
                    raise FloatingPointError("Non-finite inner metric")
                record["inner_metric"] = score
                if score < best - self.settings["min_delta"]:
                    best = score
                    stale = 0
                    self.best_epoch = epoch + 1
                    state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.network.state_dict().items()
                    }
                else:
                    stale += 1
            self.history.append(record)
            if (epoch + 1) % 25 == 0:
                print(
                    f"    epoch {epoch + 1}: loss={record['train_loss']:.5f}",
                    flush=True,
                )
            if validation is not None and stale >= self.settings["patience"]:
                break
        if state is not None:
            self.network.load_state_dict(state)
        else:
            self.best_epoch = len(self.history)
        self.network.eval()
        return self

    def _predict_tensor(self, x):
        import torch

        self.network.eval()
        with torch.no_grad():
            result = torch.cat(
                [
                    output_prediction(self.network(chunk), self.task)
                    for chunk in x.split(4096)
                ]
            )
        return result.cpu().numpy().astype(float)

    def predict(self, x):
        import torch

        a = (
            Preprocessor.apply_state(x, self.loaded_preprocessing)
            if hasattr(self, "loaded_preprocessing")
            else self.preprocessor.transform(x)
        )
        result = self._predict_tensor(torch.from_numpy(a).to(self.device))
        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite prediction")
        return result

    def predict_probability(self, x):
        import torch

        if not (self.task.startswith("hurdle") or self.task == "ziln"):
            raise ValueError("No purchase probability head")
        a = (
            Preprocessor.apply_state(x, self.loaded_preprocessing)
            if hasattr(self, "loaded_preprocessing")
            else self.preprocessor.transform(x)
        )
        self.network.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(a).to(self.device)
            p = torch.cat(
                [
                    torch.sigmoid(self.network(chunk)[:, 0])
                    for chunk in tensor.split(4096)
                ]
            )
        return p.cpu().numpy().astype(float)

    def save(self, path):
        import torch

        torch.save(
            {
                "format": 1,
                "architecture": self.architecture,
                "task": self.task,
                "settings": self.settings,
                "sequence_shape": self.sequence_shape,
                "seed": self.seed,
                "preprocessing": self.preprocessor.state(),
                "state_dict": {
                    k: v.detach().cpu() for k, v in self.network.state_dict().items()
                },
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu", threads=2):
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["format"] != 1:
            raise ValueError("Unknown neural checkpoint")
        model = cls(
            state["architecture"],
            state["task"],
            state["settings"],
            state["seed"],
            device,
            threads,
            state["sequence_shape"],
        )
        model.loaded_preprocessing = state["preprocessing"]
        model.network = make_network(
            len(state["preprocessing"]["mean"]),
            model.architecture,
            model.task,
            model.settings,
            model.sequence_shape,
        ).to(device)
        model.network.load_state_dict(state["state_dict"])
        model.network.eval()
        return model
