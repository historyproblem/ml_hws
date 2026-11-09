"""Fixed model configurations; predictions are in log1p spending units."""
import os
import random
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
        self.settings, self.seed, self.device, self.threads = settings, seed, device, threads
        self.preprocessor = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
        )

    def fit(self, x, z):
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
        a = self.preprocessor.fit_transform(x).astype(np.float32)
        xt = torch.from_numpy(a).to(self.device)
        yt = torch.from_numpy(np.asarray(z, dtype=np.float32)).to(self.device)
        layers, previous = [], a.shape[1]
        for hidden in self.settings["hidden_sizes"]:
            layers += [torch.nn.Linear(previous, hidden), torch.nn.ReLU()]
            previous = hidden
        layers.append(torch.nn.Linear(previous, 1))
        self.network = torch.nn.Sequential(*layers).to(self.device)
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.settings["learning_rate"],
            weight_decay=self.settings["weight_decay"],
        )
        self.history = []
        self.network.train()
        for epoch in range(self.settings["epochs"]):
            permutation = torch.randperm(len(xt), device=self.device)
            total = 0.0
            for indices in permutation.split(self.settings["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                predicted = self.network(xt[indices]).squeeze(-1)
                loss = torch.nn.functional.mse_loss(predicted, yt[indices])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            self.history.append({"epoch": epoch + 1, "train_log_mse": total / len(xt)})
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"MLP epoch {epoch + 1}: train log-MSE={total / len(xt):.5f}", flush=True)
        self.network.eval()
        return self

    def predict(self, x):
        import torch
        a = self.preprocessor.transform(x).astype(np.float32)
        with torch.no_grad():
            result = self.network(torch.from_numpy(a).to(self.device)).squeeze(-1)
        return result.cpu().numpy().astype(float)


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
