"""Small HW4 neural baselines; identifiers and future snapshots are excluded."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn

from .features import INSPECTION_TYPES, RESULTS, _codes


def build_sequences(rows: pd.DataFrame, history: pd.DataFrame, length: int = 16):
    """Visit tokens, oldest first, scoped to each row's snapshot and cutoff.

    Codes are an unordered set within a visit. A visit contains opaque code
    indicators, outcome/type one-hot (including unknown), age/gap and code count.
    All supplied history is observable before prediction, so bidirectional
    attention over it is appropriate. A CLS token handles an empty history.
    """
    if length < 1:
        raise ValueError("Sequence length must be positive")
    keys = rows[["forecast_period", "site_id", "history_before"]].reset_index(drop=True).copy()
    keys["_row"] = np.arange(len(rows))
    columns = ["forecast_period", "site_id", "inspection_week", "inspection_type", "result", "violation_codes"]
    h = keys.merge(history[columns], on=["forecast_period", "site_id"], how="inner")
    h["_date"] = pd.to_datetime(h.inspection_week, errors="raise")
    h["_cutoff"] = pd.to_datetime(h.history_before, errors="raise")
    h = h.loc[h._date < h._cutoff].copy()
    for c in ["inspection_type", "result", "violation_codes"]:
        h[c] = h[c].fillna("").astype(str)
    h = h.sort_values(["_row", "_date", "inspection_type", "result", "violation_codes"], kind="stable")
    h["_gap"] = h.groupby("_row")._date.diff().dt.days.fillna(0)
    h["_age"] = (h._cutoff - h._date).dt.days
    n_features = 44 + len(RESULTS) + 1 + len(INSPECTION_TYPES) + 1 + 4
    values = np.zeros((len(rows), length, n_features), dtype=np.float32)
    valid = np.zeros((len(rows), length), dtype=bool)
    counts = h.groupby("_row").size().reindex(range(len(rows)), fill_value=0).to_numpy()
    for row, group in h.groupby("_row", sort=False):
        for slot, (_, visit) in enumerate(group.tail(length).iterrows()):
            x = values[row, slot]
            codes = _codes(visit.violation_codes)
            if codes:
                x[np.asarray(codes) - 1] = 1
            x[44 + (RESULTS.index(visit.result) if visit.result in RESULTS else len(RESULTS))] = 1
            start = 44 + len(RESULTS) + 1
            x[start + (INSPECTION_TYPES.index(visit.inspection_type) if visit.inspection_type in INSPECTION_TYPES else len(INSPECTION_TYPES))] = 1
            x[-4:] = [np.log1p(visit._age) / 8, np.log1p(visit._gap) / 8,
                       len(codes) / 44, float(not visit.violation_codes.strip())]
            valid[row, slot] = True
    return values, valid, {"max_visits": int(counts.max(initial=0)),
                           "truncated_rows": int((counts > length).sum()),
                           "empty_rows": int((counts == 0).sum()), "token_features": n_features}


class TabularTransform:
    """Training-only imputation, scaling, missing flags and category vocabulary."""

    def fit(self, frame, categorical):
        self.columns = list(frame)
        self.categorical = list(categorical)
        self.numeric = [c for c in frame if c not in categorical]
        a = frame[self.numeric].to_numpy(dtype=float)
        a[~np.isfinite(a)] = np.nan
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(a)
        self.scaler = StandardScaler().fit(self.imputer.transform(a))
        self.vocabs = {c: {v: i + 1 for i, v in enumerate(sorted(frame[c].fillna("<none>").astype(str).unique()))}
                       for c in categorical}
        self.cardinalities = [len(self.vocabs[c]) + 1 for c in categorical]
        return self

    def transform(self, frame):
        if list(frame) != self.columns:
            raise ValueError("Feature schema mismatch")
        a = frame[self.numeric].to_numpy(dtype=float)
        a[~np.isfinite(a)] = np.nan
        missing = np.isnan(a)
        values = np.clip(self.scaler.transform(self.imputer.transform(a)), -8, 8)
        categories = np.column_stack([frame[c].fillna("<none>").astype(str).map(self.vocabs[c]).fillna(0)
                                      for c in self.categorical]).astype(np.int64)
        return np.concatenate([values, missing], axis=1).astype(np.float32), categories


class InspectionNet(nn.Module):
    """MLP control, visit-transformer hybrid, or an FT-style feature transformer.

    FT uses separate affine numeric tokens and categorical embeddings, then a
    pre-norm transformer. It is a compact adaptation, not the reference FT code.
    """

    def __init__(self, n_numeric, cardinalities, n_event_features, kind="mlp", width=64, length=16):
        super().__init__()
        if kind not in {"mlp", "visit_transformer", "feature_transformer"}:
            raise ValueError(kind)
        self.kind = kind
        embedding_width = width if kind == "feature_transformer" else 8
        self.embeddings = nn.ModuleList([nn.Embedding(n, embedding_width) for n in cardinalities])
        tab_dim = n_numeric + len(cardinalities) * embedding_width
        if kind != "mlp":
            self.cls = nn.Parameter(torch.empty(1, 1, width))
            nn.init.normal_(self.cls, std=.02)
            layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=.1,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        if kind == "visit_transformer":
            self.event_projection = nn.Linear(n_event_features, width)
            self.position = nn.Parameter(torch.empty(1, length, width))
            nn.init.normal_(self.position, std=.02)
            tab_dim += width
        elif kind == "feature_transformer":
            self.numeric_weight = nn.Parameter(torch.empty(n_numeric, width))
            self.numeric_bias = nn.Parameter(torch.zeros(n_numeric, width))
            nn.init.normal_(self.numeric_weight, std=.02)
            tab_dim = width
        self.head = nn.Sequential(nn.LayerNorm(tab_dim), nn.Linear(tab_dim, 128), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(128, 64), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(64, 1))

    def forward(self, numeric, categorical, events, valid):
        categories = [e(categorical[:, j]) for j, e in enumerate(self.embeddings)]
        cls = self.cls.expand(len(numeric), -1, -1) if self.kind != "mlp" else None
        if self.kind == "feature_transformer":
            tokens = numeric.unsqueeze(-1) * self.numeric_weight + self.numeric_bias
            tokens = torch.cat([cls, tokens, torch.stack(categories, dim=1)], dim=1)
            hidden = self.encoder(tokens)[:, 0]
        else:
            hidden = torch.cat([numeric, *categories], dim=1)
            if self.kind == "visit_transformer":
                tokens = torch.cat([cls, self.event_projection(events) + self.position], dim=1)
                mask = torch.cat([torch.zeros((len(valid), 1), dtype=torch.bool, device=valid.device), ~valid], dim=1)
                encoded = self.encoder(tokens, src_key_padding_mask=mask)[:, 0]
                hidden = torch.cat([hidden, encoded], dim=1)
        return self.head(hidden).squeeze(-1)
