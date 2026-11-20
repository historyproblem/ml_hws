"""Fixed legacy Visit Transformer fitting for the frozen HW4 delivery.

Epoch selection and the fresh fixed-epoch refit are separate functions. The
caller supplies temporal splits and already causal 16-visit sequences; no
outer-validation or test target is accepted by prediction or fixed refitting.
"""
from __future__ import annotations

import os
import random
import time
from numbers import Integral

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.nn.attention import SDPBackend, sdpa_kernel

from .neural import InspectionNet, TabularTransform


OPTIMIZER = dict(name="AdamW", lr=.001, weight_decay=.01,
                 gradient_clip_norm=1.0, loss="binary_cross_entropy_with_logits")
FORBIDDEN_COLUMNS = frozenset({"target", "row_id", "site_id", "forecast_period", "history_before"})


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _frame(x, cats=None):
    if not isinstance(x, pd.DataFrame) or not x.columns.is_unique:
        raise ValueError("Features must be a DataFrame with unique columns")
    if FORBIDDEN_COLUMNS.intersection(x.columns):
        raise ValueError("Targets and raw identifiers cannot enter the legacy model")
    if cats is not None and (len(cats) != len(set(cats)) or not set(cats) <= set(x)):
        raise ValueError("Categorical columns must be unique feature names")


def _setup(seed, device, threads):
    threads = _positive_integer(threads, "threads")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Legacy delivery supports cpu or cuda explicitly")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return device


def _target(y, n, device):
    value = np.asarray(y, dtype=np.float32)
    if value.shape != (n,) or not np.isfinite(value).all() or not np.isin(value, [0, 1]).all():
        raise ValueError("Targets must be row-aligned finite binary values")
    return torch.as_tensor(value, device=device)


def _data(preprocessing, x, events, valid, device):
    _frame(x)
    events = np.asarray(events, dtype=np.float32)
    valid = np.asarray(valid)
    if events.shape != (len(x), 16, 60) or valid.shape != (len(x), 16) or valid.dtype != np.bool_:
        raise ValueError("Expected events (rows,16,60) and a boolean (rows,16) validity mask")
    # Official padding is already zero. Replacing it before the legacy event
    # projection also prevents arbitrary or NaN padding from entering attention.
    events = np.where(valid[..., None], events, np.float32(0))
    if not np.isfinite(events).all():
        raise ValueError("Valid historical events must be finite")
    numeric, categorical = preprocessing.transform(x.copy(deep=True))
    if not np.isfinite(numeric).all():
        raise ValueError("Prepared numeric features must be finite")
    return tuple(torch.as_tensor(a, device=device) for a in [numeric, categorical, events, valid])


@torch.no_grad()
def _infer(model, data, batch_size):
    model.eval()
    result = []
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, len(data[0]), batch_size):
            logits = model(*(a[start:start + batch_size] for a in data))
            result.append(logits.sigmoid().cpu().numpy())
    prediction = np.concatenate(result)
    if (prediction.shape != (len(data[0]),) or not np.isfinite(prediction).all()
            or (prediction < 0).any() or (prediction > 1).any()):
        raise FloatingPointError("Invalid legacy model probabilities")
    return prediction


def _fit(xfit, yfit, cats, events, valid, *, epochs, seed, device, batch_size,
         threads, log, xeval=None, yeval=None, eval_events=None, eval_valid=None,
         patience=6):
    epochs = _positive_integer(epochs, "epochs")
    batch_size = _positive_integer(batch_size, "batch_size")
    patience = _positive_integer(patience, "patience")
    if log is not None and not callable(log):
        raise ValueError("log must be callable")
    selecting = xeval is not None
    if selecting != (yeval is not None):
        raise ValueError("Inner evaluation features and targets must be paired")
    if not len(xfit) or selecting and not len(xeval):
        raise ValueError("Training and inner evaluation splits must be nonempty")
    cats = list(cats)
    _frame(xfit, cats)
    device = _setup(seed, device, threads)
    emit = log if log is not None else lambda event, **details: None
    emit("preprocessing_start", train_rows=len(xfit), device="cpu")
    preprocessing = TabularTransform().fit(xfit.copy(deep=True), cats)
    data = _data(preprocessing, xfit, events, valid, device)
    target = _target(yfit, len(xfit), device)
    eval_data = _data(preprocessing, xeval, eval_events, eval_valid, device) if selecting else None
    eval_target = _target(yeval, len(xeval), device).cpu().numpy() if selecting else None
    kwargs = dict(n_numeric=data[0].shape[1], cardinalities=preprocessing.cardinalities,
                  n_event_features=60, kind="visit_transformer", width=64, length=16)
    model = InspectionNet(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    emit("fit_start", train_rows=len(xfit), max_epochs=epochs,
         model_device=str(next(model.parameters()).device), selecting_epochs=selecting)
    rng = np.random.default_rng(seed)
    best_ap, best_epoch = -1., 1
    curve = []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        emit("epoch_start", epoch=epoch)
        model.train()
        order = rng.permutation(len(xfit))
        total_loss = 0.
        with sdpa_kernel(SDPBackend.MATH):
            for start in range(0, len(order), batch_size):
                index = torch.as_tensor(order[start:start + batch_size], device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(*(a[index] for a in data))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[index])
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite legacy training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total_loss += loss.item() * len(index)
                if epoch == 1 and start == 0:
                    emit("first_batch_complete", logits_device=str(logits.device), loss_device=str(loss.device))
        item = dict(epoch=epoch, train_loss=total_loss / len(order))
        if selecting:
            ap = float(average_precision_score(eval_target, _infer(model, eval_data, batch_size)))
            item["inner_ap"] = ap
            if ap > best_ap:
                best_ap, best_epoch = ap, epoch
        curve.append(item)
        emit("epoch_complete", **item, epoch_seconds=round(time.monotonic() - started, 3))
        if selecting and epoch - best_epoch >= patience:
            break
    return dict(model=model, preprocessing=preprocessing,
                epochs=int(best_epoch if selecting else epochs),
                inner_curve=curve if selecting else [], training_curve=curve,
                model_kwargs=kwargs, seed=int(seed), batch_size=batch_size,
                threads=int(threads), device=str(device), optimizer=dict(OPTIMIZER),
                attention_backend="math_only", prediction_dtype="float32")


def choose_visit_epochs(xfit, yfit, xeval, yeval, cats, fit_events, fit_valid,
                        eval_events, eval_valid, *, seed=42, device="cpu",
                        max_epochs=35, patience=6, batch_size=256, threads=4,
                        log=None):
    """Choose earliest maximum-AP epoch on an earlier inner split; discard fit."""
    bundle = _fit(xfit, yfit, cats, fit_events, fit_valid,
                  epochs=max_epochs, seed=seed, device=device, batch_size=batch_size,
                  threads=threads, log=log, xeval=xeval, yeval=yeval,
                  eval_events=eval_events, eval_valid=eval_valid, patience=patience)
    return bundle["epochs"], bundle["inner_curve"]


def fit_visit(xfit, yfit, cats, events, valid, *, epochs, seed=42,
              device="cpu", batch_size=256, threads=4, log=None):
    """Fresh fixed-epoch refit; accepts no validation/test labels or eval data."""
    return _fit(xfit, yfit, cats, events, valid, epochs=epochs, seed=seed,
                device=device, batch_size=batch_size, threads=threads, log=log)


def predict_visit(bundle, x, events, valid, *, batch_size=None):
    """Class-1 probabilities using the saved transformer and training schema."""
    batch_size = _positive_integer(bundle.get("batch_size", 256) if batch_size is None else batch_size,
                                   "batch_size")
    _frame(x)
    if list(x) != bundle["preprocessing"].columns:
        raise ValueError("Feature schema mismatch")
    if not len(x):
        return np.empty(0, dtype=np.float32)
    device = next(bundle["model"].parameters()).device
    data = _data(bundle["preprocessing"], x, events, valid, device)
    return _infer(bundle["model"], data, batch_size)
