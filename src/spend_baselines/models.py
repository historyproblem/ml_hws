"""Fixed model configurations; predictions are in log1p spending units."""
import os
import random
from copy import deepcopy
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def rmsle_from_logs(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("Invalid nonnegative target")
    if not np.isfinite(prediction).all() or np.any(prediction < 0):
        raise ValueError("Invalid nonnegative log prediction")
    return float(np.sqrt(np.mean((np.log1p(target) - prediction) ** 2)))


class TorchMLP:
    def __init__(self, settings, seed=42, device="cpu", threads=2):
        self.settings = deepcopy(settings)
        self.seed, self.device, self.threads = seed, device, threads
        if not 0 <= settings.get("dropout", 0.0) < 1:
            raise ValueError("dropout must be in [0, 1)")
        if settings["epochs"] < 1 or settings["batch_size"] < 1 or threads < 1:
            raise ValueError("epochs, batch size and threads must be positive")
        self.preprocessor = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
        )

        self.loaded_preprocessing = None

    def _make_network(self, n_features):
        import torch
        layers, previous = [], n_features
        for hidden in self.settings["hidden_sizes"]:
            layers += [torch.nn.Linear(previous, hidden), torch.nn.ReLU()]
            if self.settings.get("dropout", 0.0):
                layers.append(torch.nn.Dropout(self.settings["dropout"]))
            previous = hidden
        layers.append(torch.nn.Linear(previous, 1))
        return torch.nn.Sequential(*layers).to(self.device)

    def fit(self, x, z, *, validation=None, early_stopping=None):
        if (validation is None) != (early_stopping is None):
            raise ValueError("Early stopping needs both validation data and its settings")
        if early_stopping is not None:
            if early_stopping["patience"] < 1 or early_stopping["min_delta"] < 0:
                raise ValueError("Invalid early-stopping patience/min_delta")
        # Set before importing torch/initializing CUDA for deterministic GEMM.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this environment")
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True)
        self.loaded_preprocessing = None
        a = self.preprocessor.fit_transform(x).astype(np.float32)
        self.n_features_in_ = np.asarray(x).shape[1]
        xt = torch.from_numpy(a).to(self.device)
        yt = torch.from_numpy(np.asarray(z, dtype=np.float32)).to(self.device)
        self.network = self._make_network(a.shape[1])
        if validation is not None:
            vx, vz = validation
            vz = np.asarray(vz, dtype=float)
            if len(vx) == 0 or len(vx) != len(vz) or not np.isfinite(vz).all():
                raise ValueError("Invalid early-stopping validation data")
            va = self.preprocessor.transform(vx).astype(np.float32)
            vxt = torch.from_numpy(va).to(self.device)
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.settings["learning_rate"],
            weight_decay=self.settings["weight_decay"],
        )
        self.history = []
        self.best_epoch_ = 0
        self.best_validation_rmsle_ = None
        self.stop_reason_ = "fixed_epochs" if validation is None else "max_epochs"
        best_state, best_score, stale = None, np.inf, 0
        for epoch in range(self.settings["epochs"]):
            self.network.train()
            permutation = torch.randperm(len(xt), device=self.device)
            total = 0.0
            for indices in permutation.split(self.settings["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                predicted = self.network(xt[indices]).squeeze(-1)
                loss = torch.nn.functional.mse_loss(predicted, yt[indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError("MLP training loss became non-finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            record = {"epoch": epoch + 1, "train_log_mse": total / len(xt)}
            if validation is not None:
                self.network.eval()
                with torch.no_grad():
                    vp = self.network(vxt).squeeze(-1).cpu().numpy().astype(float)
                score = float(np.sqrt(np.mean((vz - np.maximum(vp, 0.0)) ** 2)))
                if not np.isfinite(score):
                    raise FloatingPointError("MLP validation score became non-finite")
                record["validation_rmsle"] = score
                if score < best_score - early_stopping["min_delta"]:
                    best_score, stale = score, 0
                    self.best_epoch_ = epoch + 1
                    self.best_validation_rmsle_ = score
                    best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                else:
                    stale += 1
            self.history.append(record)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                extra = f", inner RMSLE={score:.5f}, best epoch={self.best_epoch_}" if validation is not None else ""
                print(f"MLP epoch {epoch + 1}: train log-MSE={total / len(xt):.5f}{extra}", flush=True)
            if validation is not None and stale >= early_stopping["patience"]:
                self.stop_reason_ = "patience_reached"
                break
        self.epochs_run_ = len(self.history)
        if best_state is not None:
            self.network.load_state_dict(best_state)
        else:
            self.best_epoch_ = self.epochs_run_
        self.network.eval()
        return self

    def predict(self, x):
        import torch
        if self.loaded_preprocessing is None:
            a = self.preprocessor.transform(x).astype(np.float32)
        else:
            p = self.loaded_preprocessing
            x = np.asarray(x, dtype=float)
            if x.ndim != 2 or x.shape[1] != self.n_features_in_:
                raise ValueError("Checkpoint feature count does not match input")
            missing = np.isnan(x)
            a = np.where(missing, p["imputer_statistics"], x)
            a = np.c_[a, missing[:, p["indicator_features"]].astype(float)]
            a = ((a - np.asarray(p["scaler_mean"])) / np.asarray(p["scaler_scale"])).astype(np.float32)
        with torch.no_grad():
            result = self.network(torch.from_numpy(a).to(self.device)).squeeze(-1)
        return result.cpu().numpy().astype(float)

    def save(self, path, *, metadata=None):
        """Store portable tensors and numeric preprocessing state, without dataset rows."""
        import torch
        if self.loaded_preprocessing is None:
            imputer, scaler = self.preprocessor.steps[0][1], self.preprocessor.steps[1][1]
            preprocessing = {"imputer_statistics": imputer.statistics_.tolist(),
                             "indicator_features": imputer.indicator_.features_.tolist(),
                             "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}
        else:
            preprocessing = self.loaded_preprocessing
        torch.save({"format_version": 1, "settings": self.settings, "seed": self.seed,
                    "n_features_in": self.n_features_in_, "preprocessing": preprocessing,
                    "state_dict": {k: v.detach().cpu() for k, v in self.network.state_dict().items()},
                    "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path, *, device="cpu", threads=2):
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported MLP checkpoint format")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this environment")
        model = cls(checkpoint["settings"], checkpoint["seed"], device, threads)
        model.n_features_in_ = checkpoint["n_features_in"]
        model.loaded_preprocessing = checkpoint["preprocessing"]
        model.metadata_ = checkpoint["metadata"]
        n = len(model.loaded_preprocessing["scaler_mean"])
        model.network = model._make_network(n)
        model.network.load_state_dict(checkpoint["state_dict"])
        model.network.eval()
        return model


def fit_model(name, x, target, config, *, device="cpu", threads=2):
    if len(x) == 0:
        raise ValueError("Cannot fit on an empty training split")
    z = np.log1p(np.asarray(target, dtype=float))
    if name == "ridge":
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(), Ridge(alpha=config["ridge"]["alpha"]),
        )
    elif name == "forest":
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            RandomForestRegressor(**config["forest"], random_state=config["seed"], n_jobs=threads),
        )
    elif name == "mlp":
        model = TorchMLP(config["mlp"], seed=config["seed"], device=device, threads=threads)
    else:
        raise ValueError(f"Unknown fitted model: {name}")
    return model.fit(x, z)


def predict_logs(name, fitted, x, features):
    if name == "zero":
        result = np.zeros(len(features))
    elif name == "logmean":
        result = features.logmean_prediction.to_numpy(dtype=float)
    else:
        result = np.asarray(fitted.predict(x), dtype=float)
    result = np.maximum(result, 0.0)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite model prediction")
    return result
