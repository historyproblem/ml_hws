"""Shared supervised optimization for fixed HW4 residual neural hypotheses.

Only an inner epoch-selection run receives evaluation inputs and labels. The
outer refit receives neither and trains for exactly the selected epoch count.
Call ``seed_all`` before constructing a model. ``train_epochs`` deliberately
does not reseed model/dropout RNG after initialization; its minibatch ordering
has a separate seeded NumPy generator. All attention uses math-only SDPA.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn.attention import SDPBackend, sdpa_kernel

COMMON_OPTIMIZER = {
    "name": "AdamW", "lr": .001, "weight_decay": .01,
    "gradient_clip_norm": 1.0,
    "loss": "binary_cross_entropy_with_logits", "class_weights": None,
}


def seed_all(seed):
    """Seed initialization/dropout RNGs; call immediately before model creation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_data(model, data):
    if not isinstance(data, (tuple, list)) or len(data) != 5:
        raise ValueError("Data must contain numeric, categories, basis, events and valid tensors")
    if not all(isinstance(value, torch.Tensor) for value in data):
        raise ValueError("Every data field must be a tensor")
    numeric, categorical, basis, events, valid = data
    if (numeric.ndim != 2 or categorical.ndim != 2 or basis.ndim != 3
            or events.ndim != 3 or valid.ndim != 2):
        raise ValueError("Data tensor dimensions must be 2,2,3,3,2 (basis: row, numeric feature, bin)")
    n = len(numeric)
    if n == 0 or any(len(value) != n for value in data):
        raise ValueError("Data tensors must have the same nonzero row count")
    if events.shape[:2] != valid.shape:
        raise ValueError("Event slots and validity mask must align")
    if not all(value.is_floating_point() for value in [numeric, basis, events]):
        raise ValueError("Numeric, basis and event inputs must be floating point")
    if categorical.dtype != torch.long or valid.dtype != torch.bool:
        raise ValueError("Categories must be int64 and event validity must be bool")
    if (categorical < 0).any().item():
        raise ValueError("Category indices must be nonnegative")
    if not all(torch.isfinite(value).all().item() for value in [numeric, basis, events]):
        raise ValueError("Prepared numeric, basis and event inputs must be finite")
    devices = {value.device for value in data}
    devices.update(value.device for value in model.parameters())
    devices.update(value.device for value in model.buffers())
    if len(devices) != 1:
        raise ValueError("Model and all input tensors must share one device")
    return n, numeric.device


def _validate_targets(targets, n, device, name):
    if not isinstance(targets, torch.Tensor) or targets.ndim != 1 or len(targets) != n:
        raise ValueError(f"{name} must be a row-aligned one-dimensional tensor")
    if targets.device != device or targets.is_complex():
        raise ValueError(f"{name} must be real and on the model device")
    if not torch.isfinite(targets).all().item() or not ((targets == 0) | (targets == 1)).all().item():
        raise ValueError(f"{name} must contain finite binary labels")


def _checked_logits(model, batch):
    logits = model(*batch)
    if not isinstance(logits, torch.Tensor) or logits.shape != (len(batch[0]),):
        raise ValueError("Model must return one logit per row")
    if logits.device != batch[0].device or not logits.is_floating_point():
        raise ValueError("Logits must be floating point on the input device")
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError("Nonfinite model logits")
    return logits


def _memory(device):
    return round(torch.cuda.memory_allocated(device) / 1024 ** 2, 2) if device.type == "cuda" else 0.0


@torch.no_grad()
def infer(model, data, batch_size=256):
    """Return finite class-1 probabilities in row order, leaving model in eval."""
    _positive_integer(batch_size, "batch_size")
    n, _ = _validate_data(model, data)
    model.eval()
    output = []
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, n, batch_size):
            batch = tuple(value[start:start + batch_size] for value in data)
            output.append(_checked_logits(model, batch).sigmoid().cpu().numpy())
    result = np.concatenate(output)
    if result.shape != (n,) or not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise FloatingPointError("Invalid class-1 probabilities")
    return result


def train_epochs(model, fit_data, targets, *, epochs, batch_size, seed,
                 eval_data=None, eval_targets=None, patience=10, log=None):
    """Return ``(chosen_epochs, curve)`` using ordinary BCE and fixed AdamW.

    When an inner evaluation pair is supplied, select the earliest epoch with
    maximum AP and stop after ``patience`` consecutive epochs without a new
    maximum. This does not restore best weights: the inner model is discarded
    and the caller initializes a new outer model for ``chosen_epochs`` epochs.
    Without that pair, no evaluation or AP call occurs and exactly ``epochs``
    epochs run. Model creation and global RNG seeding belong to the caller.
    """
    for name, value in [("epochs", epochs), ("batch_size", batch_size), ("patience", patience)]:
        _positive_integer(value, name)
    if (eval_data is None) != (eval_targets is None):
        raise ValueError("Inner evaluation data and targets must be supplied together")
    if log is not None and not callable(log):
        raise ValueError("log must be callable")
    n, device = _validate_data(model, fit_data)
    _validate_targets(targets, n, device, "Training targets")
    selecting = eval_data is not None
    evaluation_y = None
    if selecting:
        eval_n, eval_device = _validate_data(model, eval_data)
        if eval_device != device:
            raise ValueError("Training and inner evaluation data must share one device")
        _validate_targets(eval_targets, eval_n, device, "Inner evaluation targets")
        evaluation_y = eval_targets.detach().cpu().numpy()
    emit = log if log is not None else lambda event, **fields: None
    optimizer = torch.optim.AdamW(model.parameters(), lr=COMMON_OPTIMIZER["lr"],
                                  weight_decay=COMMON_OPTIMIZER["weight_decay"])
    rng = np.random.default_rng(seed)
    best_ap, best_epoch = -1.0, 1
    curve = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        emit("epoch_start", epoch=epoch, max_epochs=int(epochs), train_rows=n,
             cuda_allocated_mib=_memory(device))
        order = rng.permutation(n)
        total_loss = 0.0
        with sdpa_kernel(SDPBackend.MATH):
            for start in range(0, n, batch_size):
                indices = torch.as_tensor(order[start:start + batch_size], device=device)
                batch = tuple(value[indices] for value in fit_data)
                optimizer.zero_grad(set_to_none=True)
                logits = _checked_logits(model, batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets[indices].to(dtype=logits.dtype))
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                COMMON_OPTIMIZER["gradient_clip_norm"],
                                                error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.item() * len(indices)
                if epoch == 1 and start == 0:
                    emit("first_batch", epoch=epoch, loss=float(loss.item()),
                         logits_device=str(logits.device), loss_device=str(loss.device),
                         seconds=time.perf_counter() - started,
                         cuda_allocated_mib=_memory(device))
        row = {"epoch": epoch, "train_loss": total_loss / n}
        if selecting:
            row["inner_ap"] = float(average_precision_score(
                evaluation_y, infer(model, eval_data, batch_size)))
            if row["inner_ap"] > best_ap:
                best_ap, best_epoch = row["inner_ap"], epoch
        row["seconds"] = time.perf_counter() - started
        curve.append(row)
        emit("epoch_complete", **row, cuda_allocated_mib=_memory(device))
        if selecting and epoch - best_epoch >= patience:
            break
    return int(best_epoch if selecting else epochs), curve
