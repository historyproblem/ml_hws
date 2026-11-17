"""Grouped masked-feature denoising and a shared supervised control.

This is a task-specific model, not a reproduction of VIME or a language BERT.
The pretraining loss accepts no labels. Historical windows and their derived
code/outcome fields are masked together, including natural-missing indicators.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.nn import functional as F

from .neural import TabularTransform


def feature_group(name):
    static = {"facility_type": "static_facility", "risk_level": "static_risk", "area_id": "static_area"}
    if name in static:
        return static[name]
    if name.startswith("ctx_"):
        for context in ["area_facility", "facility_risk", "area_risk", "facility", "area", "risk"]:
            if name.startswith(f"ctx_{context}_"):
                return "context_" + context
        raise ValueError(f"Unknown context field: {name}")
    if name in {"forecast_year", "forecast_quarter", "cutoff_gap_days"}:
        return "calendar"
    if name.startswith("code") or "violation" in name:
        return "history_codes_burden"
    if name.endswith("_result") or any(s in name for s in ["_fail_", "_pass_", "_conditional_"]):
        return "history_outcomes"
    if any(s in name for s in ["age_days", "gap", "span"]):
        return "history_timing"
    if name.startswith("type_") or "inspection_type" in name or "reinspection" in name:
        return "history_types"
    if name.endswith("_n_visits"):
        return "history_volume"
    raise ValueError(f"Unassigned feature: {name}")


def group_schema(numeric, categorical):
    groups = {}
    for kind, columns in [("numeric", numeric), ("categorical", categorical)]:
        for i, name in enumerate(columns):
            group = groups.setdefault(feature_group(name), {"numeric": [], "categorical": []})
            group[kind].append(i)
    return [{"name": name, **value} for name, value in groups.items()]


class MaskedTransform(TabularTransform):
    """Training-only rank normalization, explicit missing flags and vocabulary."""

    def fit(self, frame, categorical):
        forbidden = {"target", "site_id", "row_id", "forecast_period", "history_before"} & set(frame)
        if forbidden:
            raise ValueError(f"Forbidden model inputs: {sorted(forbidden)}")
        super().fit(frame, categorical)
        numeric, _ = super().transform(frame)
        n = len(self.numeric)
        self.variable = np.ptp(numeric[:, :n], axis=0) > 0
        self.quantile = QuantileTransformer(n_quantiles=min(256, len(frame)), output_distribution="normal",
                                            subsample=None, random_state=42).fit(numeric[:, :n])
        self.groups = group_schema(self.numeric, self.categorical)
        return self

    def transform(self, frame):
        numeric, categories = super().transform(frame)
        n = len(self.numeric)
        values = np.clip(self.quantile.transform(numeric[:, :n]), -4, 4).astype(np.float32)
        values[:, ~self.variable] = 0
        return np.concatenate([values, numeric[:, n:]], axis=1), categories


class GroupMaskedNet(nn.Module):
    def __init__(self, n_numeric, cardinalities, groups, width=96, layers=3, dropout=.15, embedding_size=64):
        super().__init__()
        self.n_numeric, self.groups = n_numeric, groups
        self.embeddings = nn.ModuleList([nn.Embedding(n, 8, padding_idx=0) for n in cardinalities])
        self.projections = nn.ModuleList()
        self.numeric_heads = nn.ModuleDict()
        for i, group in enumerate(groups):
            size = len(group["numeric"]) * 2 + len(group["categorical"]) * 8
            self.projections.append(nn.Sequential(nn.LayerNorm(size), nn.Linear(size, width), nn.GELU(),
                                                   nn.Linear(width, width)))
            if group["numeric"]:
                self.numeric_heads[str(i)] = nn.Linear(width, 2 * len(group["numeric"]))
        self.category_heads = nn.ModuleList([nn.Linear(width, n) for n in cardinalities])
        self.group_identity = nn.Parameter(torch.empty(1, len(groups), width))
        self.mask_token = nn.Parameter(torch.empty(1, len(groups), width))
        self.cls = nn.Parameter(torch.empty(1, 1, width))
        for p in [self.group_identity, self.mask_token, self.cls]:
            nn.init.normal_(p, std=.02)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.bottleneck = nn.Linear(width, embedding_size)
        self.decoder_context = nn.Linear(embedding_size, width)
        direct = n_numeric * 2 + len(cardinalities) * 8
        self.head = nn.Sequential(nn.LayerNorm(direct + embedding_size), nn.Linear(direct + embedding_size, 128),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def expanded_mask(self, mask):
        """Mask count-derived aliases when masking visit volume."""
        mask = mask.clone()
        names = [g["name"] for g in self.groups]
        if "history_volume" in names:
            volume = mask[:, names.index("history_volume")].clone()
            for name in ["history_codes_burden", "history_outcomes", "history_types"]:
                if name in names:
                    mask[:, names.index(name)] |= volume
        return mask

    def sample_mask(self, n, probability, device):
        if not 0 < probability < 1 or len(self.groups) < 2:
            raise ValueError("Need 0<mask_probability<1 and at least two groups")
        mask = torch.rand(n, len(self.groups), device=device) < probability
        mask[~mask.any(dim=1), 0] = True
        mask = self.expanded_mask(mask)
        # Context/static/calendar groups are outside volume dependencies.
        visible = next((i for i, g in enumerate(self.groups) if not g["name"].startswith("history_")), None)
        if visible is None:
            raise ValueError("A non-history group is needed as a visible anchor")
        mask[mask.all(dim=1), visible] = False
        return mask

    def encode(self, numeric, categorical, mask=None):
        categories = [e(categorical[:, j]) for j, e in enumerate(self.embeddings)]
        tokens = []
        for projection, group in zip(self.projections, self.groups):
            idx = group["numeric"]
            pieces = [numeric[:, idx + [i + self.n_numeric for i in idx]]]
            pieces.extend(categories[j] for j in group["categorical"])
            tokens.append(projection(torch.cat(pieces, dim=1)))
        tokens = torch.stack(tokens, dim=1)
        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), self.mask_token, tokens)
        tokens = tokens + self.group_identity
        encoded = self.norm(self.encoder(torch.cat([self.cls.expand(len(numeric), -1, -1), tokens], dim=1)))
        embedding = self.bottleneck(encoded[:, 0])
        direct = torch.cat([numeric, *categories], dim=1)
        return embedding, encoded[:, 1:], direct

    def forward(self, numeric, categorical):
        embedding, _, direct = self.encode(numeric, categorical)
        return self.head(torch.cat([direct, embedding], dim=1)).squeeze(-1)

    def pretraining_loss(self, numeric, categorical, mask):
        mask = self.expanded_mask(mask)
        embedding, hidden, _ = self.encode(numeric, categorical, mask)
        hidden = hidden + self.decoder_context(embedding).unsqueeze(1)
        losses, active = [], []
        for i, group in enumerate(self.groups):
            terms = []
            idx = group["numeric"]
            if idx:
                predicted, missing_logits = self.numeric_heads[str(i)](hidden[:, i]).chunk(2, dim=1)
                missing = numeric[:, [j + self.n_numeric for j in idx]]
                observed = 1 - missing
                errors = F.smooth_l1_loss(predicted, numeric[:, idx], reduction="none")
                terms.append((errors * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1))
                terms.append(.2 * F.binary_cross_entropy_with_logits(missing_logits, missing, reduction="none").mean(dim=1))
            for j in group["categorical"]:
                logits = self.category_heads[j](hidden[:, i])
                terms.append(F.cross_entropy(logits, categorical[:, j], reduction="none") / max(math.log(logits.shape[1]), 1))
            per_row = torch.stack(terms).sum(dim=0)
            weight = mask[:, i].to(per_row.dtype)
            losses.append((per_row * weight).sum() / weight.sum().clamp_min(1))
            active.append((weight.sum() > 0).to(per_row.dtype))
        return torch.stack(losses).sum() / torch.stack(active).sum().clamp_min(1)
