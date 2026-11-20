"""Official TabM models with training-only preprocessing and member-wise BCE.

No pretrained weights or target-aware binning are used. Numeric imputation,
scaling, missing flags, category vocabularies and PLE boundaries are fitted only
to the caller's training frame. All ensemble members receive their own loss;
only inference averages their probabilities.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
import warnings

import numpy as np
import pandas as pd
import rtdl_num_embeddings
import tabm
import torch

from .neural import TabularTransform


FORBIDDEN_COLUMNS = frozenset({"target", "row_id", "site_id", "forecast_period", "history_before"})


def _positive_integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _check_frame(frame):
    if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
            or not all(isinstance(c, str) for c in frame.columns)):
        raise ValueError("Features must be a DataFrame with unique string columns")
    if FORBIDDEN_COLUMNS.intersection(frame.columns):
        raise ValueError("Targets and raw identifiers are forbidden features")


def _single_bin_warning_filter():
    # Binary indicators and constant fallbacks legitimately use one PLE segment.
    # Record them in preprocessing/model metadata instead of emitting hundreds
    # of identical upstream informational warnings for missing-flag features.
    warnings.filterwarnings(
        "ignore", message=r"The \d+-th feature has just two bin edges.*",
        category=UserWarning, module=r"rtdl_num_embeddings",
    )


class TabMPreprocessor:
    """Existing numeric/category transform plus fit-only official PLE bins.

    ``n_numeric`` counts standardized values AND their missing flags. Thus the
    346-column context view with ten categories has 672 numeric inputs. Every
    fitted column gets a separate PLE embedding, including the missing flags.
    Constant columns get deterministic [value-1,value+1] boundaries. For a
    two-row fit, nonconstant columns use their observed min/max; otherwise the
    official quantile ``compute_bins`` runs on nonconstant training columns.
    """

    def __init__(self, bins=16):
        self.n_bins = _positive_integer("bins", bins, minimum=2)

    def _base_transform(self, frame):
        if not len(frame):
            return (np.empty((0, 2 * len(self.numeric_names)), dtype=np.float32),
                    np.empty((0, len(self.categorical)), dtype=np.int64))
        if self.categorical:
            return self.base.transform(frame.copy(deep=True))
        # Preserve the existing numeric path for a numeric-only synthetic/test
        # view; TabularTransform otherwise column_stacks an empty category list.
        a = frame[self.numeric_names].to_numpy(dtype=float, copy=True)
        a[~np.isfinite(a)] = np.nan
        missing = np.isnan(a)
        scaled = np.clip(self.base.scaler.transform(self.base.imputer.transform(a)), -8, 8)
        return np.concatenate([scaled, missing], axis=1).astype(np.float32), np.empty((len(a), 0), dtype=np.int64)

    def fit(self, frame, cats):
        _check_frame(frame)
        cats = list(cats)
        if len(cats) != len(set(cats)) or not set(cats) <= set(frame):
            raise ValueError("Categories must be unique feature columns")
        numeric_names = [c for c in frame if c not in cats]
        if not len(frame) or not numeric_names:
            raise ValueError("At least one fit row and numeric feature are required")
        self.base = TabularTransform().fit(frame.copy(deep=True), cats)
        self.columns = list(frame)
        self.categorical = cats
        self.numeric_names = numeric_names
        self.cardinalities = list(self.base.cardinalities)
        numeric, _ = self._base_transform(frame)
        self.n_numeric = numeric.shape[1]
        self.numeric_input_names = numeric_names + [f"missing::{c}" for c in numeric_names]
        values = torch.as_tensor(numeric, dtype=torch.float32)
        constant = (values == values[0]).all(dim=0).numpy()
        self.constant_indices = np.flatnonzero(constant).tolist()
        self.all_missing_columns = [
            name for name in numeric_names
            if not np.isfinite(frame[name].to_numpy(dtype=float, copy=True)).any()
        ]
        varying = np.flatnonzero(~constant)
        boundaries = {}
        if len(varying) and len(frame) > 2:
            with warnings.catch_warnings():
                _single_bin_warning_filter()
                computed = rtdl_num_embeddings.compute_bins(
                    values[:, varying], n_bins=min(self.n_bins, len(frame) - 1))
            boundaries.update(zip(varying.tolist(), computed))
        elif len(varying):
            for j in varying:
                boundaries[int(j)] = torch.tensor(
                    [float(values[:, j].min()), float(values[:, j].max())], dtype=torch.float32)
        self.bins = []
        for j in range(self.n_numeric):
            if constant[j]:
                value = float(values[0, j])
                edges = torch.tensor([value - 1., value + 1.], dtype=torch.float32)
            else:
                edges = boundaries[j].detach().cpu().clone()
            if not torch.isfinite(edges).all() or len(edges) < 2 or not (edges.diff() > 0).all():
                raise ValueError("PLE boundaries must be finite and strictly increasing")
            self.bins.append(edges)
        self.single_bin_indices = [j for j, edges in enumerate(self.bins) if len(edges) == 2]
        self.bin_edges_sha256 = hashlib.sha256(json.dumps(
            [edges.tolist() for edges in self.bins], separators=(",", ":"),
            allow_nan=False).encode()).hexdigest()
        return self

    def transform(self, frame):
        _check_frame(frame)
        if not hasattr(self, "base"):
            raise ValueError("TabMPreprocessor must be fitted before transform")
        if list(frame) != self.columns:
            raise ValueError("Feature schema differs from the fitted training frame")
        numeric, categories = self._base_transform(frame)
        if not np.isfinite(numeric).all():
            raise ValueError("Prepared TabM numeric inputs must be finite")
        return numeric.astype(np.float32, copy=False), categories.astype(np.int64, copy=False)


def build_tabm(preprocessor, k=32, width=256, blocks=3, dropout=.1,
               bins=16, embedding_dim=8):
    """Build the pinned OFFICIAL TabM and PiecewiseLinearEmbeddings modules."""
    k = _positive_integer("k", k)
    width = _positive_integer("width", width)
    blocks = _positive_integer("blocks", blocks)
    bins = _positive_integer("bins", bins, minimum=2)
    embedding_dim = _positive_integer("embedding_dim", embedding_dim)
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout must be finite and in [0,1)")
    if not isinstance(preprocessor, TabMPreprocessor) or not hasattr(preprocessor, "bins"):
        raise ValueError("A fitted TabMPreprocessor is required")
    if preprocessor.n_bins != bins:
        raise ValueError("Model bins differ from the fitted training-only bin configuration")
    with warnings.catch_warnings():
        _single_bin_warning_filter()
        embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            [edges.clone() for edges in preprocessor.bins], embedding_dim,
            activation=False, version="B")
    official_kwargs = dict(
        n_num_features=preprocessor.n_numeric,
        cat_cardinalities=preprocessor.cardinalities,
        d_out=1, n_blocks=blocks, d_block=width, dropout=float(dropout),
        activation="ReLU", k=k, arch_type="tabm", start_scaling_init="normal",
    )
    model = tabm.TabM.make(num_embeddings=embeddings, **official_kwargs)
    # Flat build kwargs are sufficient to rebuild against the saved
    # preprocessing. The second dictionary documents exact upstream arguments.
    model.hw4_model_kwargs = dict(k=k, width=width, blocks=blocks,
                                  dropout=float(dropout), bins=bins, embedding_dim=embedding_dim)
    model.hw4_official_kwargs = dict(
        **official_kwargs,
        num_embeddings=dict(
            module="rtdl_num_embeddings.PiecewiseLinearEmbeddings", version="B",
            activation=False, d_embedding=embedding_dim,
            bin_edges_sha256=preprocessor.bin_edges_sha256,
            constant_indices=preprocessor.constant_indices,
            single_bin_indices=preprocessor.single_bin_indices,
        ),
    )
    return model


def member_logits(model, numeric, cats):
    """Return a strict (batch,K) float32 logit tensor, one per member."""
    if not isinstance(model, tabm.TabM):
        raise ValueError("Expected the official tabm.TabM model")
    if (not isinstance(numeric, torch.Tensor) or numeric.ndim != 2
            or numeric.dtype != torch.float32 or numeric.shape[1] != model._n_num_features):
        raise ValueError("Numeric input must be float32 (batch,numeric_features)")
    if (not isinstance(cats, torch.Tensor) or cats.ndim != 2
            or cats.dtype != torch.int64 or cats.shape != (len(numeric), model._n_cat_features)):
        raise ValueError("Category input must be int64 (batch,categorical_features)")
    if numeric.device != cats.device or numeric.device != next(model.parameters()).device:
        raise ValueError("Model and input tensors must share one device")
    if not torch.isfinite(numeric).all():
        raise ValueError("Numeric inputs must be finite")
    values = model(numeric, cats if model._n_cat_features else None)
    if values.shape != (len(numeric), model.k, 1) or values.dtype != torch.float32:
        raise ValueError("Official TabM must return float32 (batch,K,1) logits")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite TabM logits")
    return values.squeeze(-1)


def loss_tabm(logits, y):
    """Mean BCE over each member independently; never BCE of averaged logits."""
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or logits.dtype != torch.float32 or not logits.shape[0] or not logits.shape[1]):
        raise ValueError("Logits must be a nonempty float32 (batch,K) tensor")
    target = torch.as_tensor(y, device=logits.device, dtype=torch.float32)
    if target.shape != (len(logits),) or not torch.isfinite(target).all():
        raise ValueError("Targets must be a finite row-aligned binary vector")
    if not ((target == 0) | (target == 1)).all() or not torch.isfinite(logits).all():
        raise ValueError("Targets must be binary and logits finite")
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target[:, None].expand_as(logits), reduction="mean")


@torch.no_grad()
def predict_tabm(model, data, batch_size=256):
    """Row-aligned float32 mean member probabilities on the model's device."""
    batch_size = _positive_integer("batch_size", batch_size)
    if not isinstance(data, (tuple, list)) or len(data) != 2:
        raise ValueError("Data must contain numeric and categorical arrays/tensors")
    device = next(model.parameters()).device
    numeric = torch.as_tensor(data[0], device=device, dtype=torch.float32)
    cats = torch.as_tensor(data[1], device=device)
    if numeric.ndim != 2 or cats.ndim != 2 or len(cats) != len(numeric):
        raise ValueError("Prediction inputs must be row-aligned matrices")
    model.eval()
    if not len(numeric):
        return np.empty(0, dtype=np.float32)
    result = []
    for start in range(0, len(numeric), batch_size):
        logits = member_logits(model, numeric[start:start + batch_size], cats[start:start + batch_size])
        result.append(logits.sigmoid().mean(dim=1).cpu().numpy())
    probability = np.concatenate(result)
    if (probability.shape != (len(numeric),) or probability.dtype != np.float32
            or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any()):
        raise FloatingPointError("Invalid TabM probabilities")
    return probability
