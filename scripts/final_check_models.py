"""Joint-loss variants for final checks, separate from the frozen original suite."""
import os
import random
import numpy as np
from spend_baselines.suite_models import Neural, Preprocessor, make_network, output_prediction


def weighted_joint_loss(raw, y, w, coefficients):
    import torch
    f = torch.nn.functional
    positive = y > 0
    z = torch.log1p(y)

    def average(values, weights):
        return (values * weights).sum() / weights.sum().clamp_min(1e-12)

    loss = coefficients["purchase_bce"] * average(
        f.binary_cross_entropy_with_logits(raw[:, 0], positive.float(), reduction="none"), w)
    if positive.any():
        loss = loss + coefficients["positive_log_mse"] * average(
            (f.softplus(raw[positive, 1]) - z[positive]) ** 2, w[positive])
    return loss + coefficients["total_log_mse"] * average(
        (output_prediction(raw, "hurdle_joint") - z) ** 2, w)


class FinalJoint(Neural):
    """Same forward/preprocessing/checkpoint format; only explicit loss weights vary."""
    def fit(self, x, y, w, validation=None, epochs=None):
        if self.task != "hurdle_joint" or self.sequence_shape is not None:
            raise ValueError("FinalJoint is a dense joint-hurdle model")
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
        keep = w > 0
        x, y, w = x[keep], y[keep], w[keep]
        if not len(y):
            raise ValueError("Empty final-check training split")
        self.preprocessor = Preprocessor().fit(x, w)
        a = self.preprocessor.transform(x)
        self.network = make_network(a.shape[1], self.architecture, self.task, self.settings, None).to(self.device)
        tx = torch.from_numpy(a).to(self.device)
        ty = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        tw = torch.as_tensor(w, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.settings["learning_rate"],
                                     weight_decay=self.settings["weight_decay"])
        if validation is not None:
            vx, vy = validation
            if not len(vy):
                raise ValueError("Empty final-check inner validation")
            va = torch.from_numpy(self.preprocessor.transform(vx)).to(self.device)
        coefficients = self.settings["loss_weights"]
        if set(coefficients) != {"purchase_bce", "positive_log_mse", "total_log_mse"}:
            raise ValueError("Unknown joint-loss component")
        if any(not np.isfinite(v) or v < 0 for v in coefficients.values()) or sum(coefficients.values()) <= 0:
            raise ValueError("Invalid loss weights")
        self.history, self.best_epoch = [], 0
        best, state, stale = np.inf, None, 0
        for epoch in range(epochs or self.settings["max_epochs"]):
            self.network.train()
            total, mass = 0.0, 0.0
            for ix in torch.randperm(len(tx), device=self.device).split(self.settings["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                loss = weighted_joint_loss(self.network(tx[ix]), ty[ix], tw[ix], coefficients)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite final-check joint loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
                optimizer.step()
                total += float(loss.detach()) * len(ix)
                mass += len(ix)
            record = {"epoch": epoch + 1, "train_loss": total / mass}
            if validation is not None:
                predicted = self._predict_tensor(va)
                score = float(np.sqrt(np.mean((np.log1p(vy) - predicted) ** 2)))
                if not np.isfinite(score):
                    raise FloatingPointError("Non-finite final-check inner score")
                record["inner_metric"] = score
                if score < best - self.settings["min_delta"]:
                    best, stale, self.best_epoch = score, 0, epoch + 1
                    state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                else:
                    stale += 1
            self.history.append(record)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch + 1}: loss={record['train_loss']:.5f}, best={self.best_epoch}", flush=True)
            if validation is not None and stale >= self.settings["patience"]:
                break
        if state is not None:
            self.network.load_state_dict(state)
        else:
            self.best_epoch = len(self.history)
        self.network.eval()
        return self
