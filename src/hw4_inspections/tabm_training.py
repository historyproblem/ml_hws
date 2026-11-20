"""Fixed official-TabM optimization for inner selection and fresh outer refits.

An evaluation pair is used only for inner epoch selection. The returned inner
model is its last trained state and must be discarded: callers initialize a
fresh fit without evaluation data for the selected number of epochs. No AP is
computed during a fixed-epoch outer/full refit.
"""
from __future__ import annotations

import os
import random
import time
from numbers import Integral

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .tabm_models import TabMPreprocessor, build_tabm, loss_tabm, member_logits, predict_tabm


OPTIMIZER = {
    "name": "AdamW", "lr": .002, "weight_decay": .0003,
    "gradient_clip_norm": 1.0,
    "loss": "mean_binary_cross_entropy_with_logits_per_member",
    "class_weights": None, "amp": False,
}


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _seed_and_device(seed, device, threads):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 or seed >= 2 ** 32:
        raise ValueError("seed must be an integer in [0,2**32)")
    torch.set_num_threads(_positive_integer("threads", threads))
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu or cuda explicitly")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise RuntimeError("CUDA determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
        # Resolve bare 'cuda' to the actual device index before strict device
        # assertions: torch.device('cuda') != torch.device('cuda:0').
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    else:
        device = torch.device("cpu")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    return device


def _targets(y, n, device):
    values = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    if (values.shape != (n,) or not np.isfinite(values).all()
            or not np.isin(values, [0, 1]).all()):
        raise ValueError("Targets must be row-aligned finite binary labels")
    return torch.as_tensor(values, device=device, dtype=torch.float32)


def _data(preprocessor, frame, device):
    numeric, categories = preprocessor.transform(frame)
    values = (torch.as_tensor(numeric, device=device, dtype=torch.float32),
              torch.as_tensor(categories, device=device, dtype=torch.int64))
    if any(value.device != device for value in values):
        raise RuntimeError("Prepared data did not reach the requested training device")
    return values


def _memory(device):
    return round(torch.cuda.memory_allocated(device) / 1024 ** 2, 2) if device.type == "cuda" else 0.


def fit_tabm(xtrain, ytrain, cats, *, k=32, seed=42, device="cpu", epochs=60,
             patience=10, batch_size=256, width=256, blocks=3, bins=16,
             embedding_dim=8, threads=4, eval_frame=None, eval_target=None,
             log=None):
    """Fit with fixed optimization; optionally select earliest inner maximum AP.

    Preprocessing and bins are fitted only on xtrain. With no evaluation pair,
    exactly ``epochs`` epochs run and ``epochs`` is returned unchanged. An inner
    selection fit returns the selected count but does not restore best weights;
    its last-epoch model is discarded before a fresh outer/full fit.
    """
    epochs = _positive_integer("epochs", epochs)
    patience = _positive_integer("patience", patience)
    batch_size = _positive_integer("batch_size", batch_size)
    if (eval_frame is None) != (eval_target is None):
        raise ValueError("Inner evaluation features and targets must be supplied together")
    if log is not None and not callable(log):
        raise ValueError("log must be callable")
    selecting = eval_frame is not None
    if not len(xtrain) or selecting and not len(eval_frame):
        raise ValueError("Training and inner evaluation frames must be nonempty")
    device = _seed_and_device(seed, device, threads)
    emit = log if log is not None else lambda event, **details: None
    emit("preprocessing_start", train_rows=len(xtrain), preprocessing_device="cpu")
    preprocessor = TabMPreprocessor(bins=bins).fit(xtrain, cats)
    data = _data(preprocessor, xtrain, device)
    target = _targets(ytrain, len(xtrain), device)
    eval_data = _data(preprocessor, eval_frame, device) if selecting else None
    eval_y = _targets(eval_target, len(eval_frame), device).cpu().numpy() if selecting else None
    model = build_tabm(preprocessor, k=k, width=width, blocks=blocks, bins=bins,
                       embedding_dim=embedding_dim).to(device)
    if any(parameter.device != data[0].device for parameter in model.parameters()):
        raise RuntimeError("Model and prepared training tensors use different devices")
    optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMIZER["lr"],
                                  weight_decay=OPTIMIZER["weight_decay"])
    emit("fit_start", train_rows=len(xtrain), max_epochs=epochs,
         model_device=str(next(model.parameters()).device),
         tensor_devices=sorted({str(value.device) for value in [*data, target]}),
         selecting_epochs=selecting, cuda_allocated_mib=_memory(device))
    rng = np.random.default_rng(seed)
    best_ap, best_epoch = -1., 1
    curve = []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        emit("epoch_start", epoch=epoch)
        order = rng.permutation(len(xtrain))
        total_loss = 0.
        # Explicit FP32 even if a caller has an ambient autocast context.
        with torch.autocast(device_type=device.type, enabled=False):
            for start in range(0, len(order), batch_size):
                indices = torch.as_tensor(order[start:start + batch_size], device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = member_logits(model, data[0][indices], data[1][indices])
                loss = loss_tabm(logits, target[indices])
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite TabM training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                OPTIMIZER["gradient_clip_norm"],
                                                error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.item() * len(indices)
                if epoch == 1 and start == 0:
                    emit("first_batch_complete", logits_device=str(logits.device),
                         loss_device=str(loss.device), member_logits_shape=list(logits.shape),
                         loss=float(loss.item()), cuda_allocated_mib=_memory(device))
        item = {"epoch": epoch, "train_loss": total_loss / len(order)}
        if selecting:
            with torch.autocast(device_type=device.type, enabled=False):
                probability = predict_tabm(model, eval_data, batch_size=batch_size)
            item["inner_ap"] = float(average_precision_score(eval_y, probability))
            if item["inner_ap"] > best_ap:
                best_ap, best_epoch = item["inner_ap"], epoch
        curve.append(item)
        emit("epoch_complete", **item,
             epoch_seconds=round(time.monotonic() - started, 3),
             cuda_allocated_mib=_memory(device))
        if selecting and epoch - best_epoch >= patience:
            break
    model.eval()
    return {
        "model": model, "preprocessor": preprocessor, "preprocessing": preprocessor,
        "epochs": int(best_epoch if selecting else epochs),
        "trained_epochs": len(curve), "best_ap": float(best_ap) if selecting else None,
        "curve": curve, "training_curve": curve, "bestcurve": curve,
        "model_kwargs": dict(model.hw4_model_kwargs),
        "official_kwargs": dict(model.hw4_official_kwargs),
        "optimizer": dict(OPTIMIZER), "batch_size": batch_size,
        "seed": int(seed), "threads": int(threads), "device": str(device),
        "training_rows": len(xtrain), "selection_rows": len(eval_frame) if selecting else 0,
        "selected_on_inner_ap": selecting, "restored_best_weights": False,
        "prediction_dtype": "float32", "amp": False,
    }


def predict_bundle(bundle, frame, batch_size=None):
    """Predict mean member probabilities using saved fit-only preprocessing."""
    batch_size = _positive_integer("batch_size", bundle.get("batch_size", 256)
                                   if batch_size is None else batch_size)
    preprocessor = bundle.get("preprocessor")
    if preprocessor is None:
        preprocessor = bundle["preprocessing"]
    data = preprocessor.transform(frame)
    device = next(bundle["model"].parameters()).device
    with torch.autocast(device_type=device.type, enabled=False):
        return predict_tabm(bundle["model"], data, batch_size=batch_size)
