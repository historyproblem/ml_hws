"""Matched tabular residual controls, PLE embeddings and causal visit attention.

This module only defines models. Fit-only numeric knots, category dictionaries
and strict historical snapshots are prepared by the caller; no identifiers or
target-derived features enter these networks.
"""
from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import nn


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class ResidualBlock(nn.Module):
    """A pre-normalized branch; the skip flag introduces no extra parameters."""

    def __init__(self, width, dropout):
        super().__init__()
        self.branch = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * width, width), nn.Dropout(dropout),
        )

    def forward(self, x, use_skip=True):
        branch = self.branch(x)
        return x + branch if use_skip else branch


class ResidualInspectionNet(nn.Module):
    """Four fixed model hypotheses with the same categorical embedding width.

    ``numeric`` contains N scaled values followed by N natural missing flags.
    PLE variants instead learn one ``bins x embedding_dim`` matrix per numeric
    feature and append the same N missing flags. ``mlp_control`` and ``resnet``
    have identical state dictionaries and initialization under a matched seed;
    only residual additions differ.

    Visit events have exactly 60 columns: 44 opaque code indicators, four
    outcomes, eight inspection types, and four previously prepared time/load
    values. Attention sees only historical visits supplied by the caller.
    """

    KINDS = frozenset({"mlp_control", "resnet", "ple_resnet", "ple_visit_transformer"})

    def __init__(self, n_numeric, cardinalities, kind="resnet", bins=16,
                 embedding_dim=8, width=128, blocks=3, dropout=.15,
                 length=16, visit_width=64):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"Unknown residual model: {kind}")
        self.kind = kind
        self.n_numeric = _positive_integer("n_numeric", n_numeric)
        self.bins = _positive_integer("bins", bins)
        self.embedding_dim = _positive_integer("embedding_dim", embedding_dim)
        self.width = _positive_integer("width", width)
        blocks = _positive_integer("blocks", blocks)
        self.length = _positive_integer("length", length)
        self.visit_width = _positive_integer("visit_width", visit_width)
        self.cardinalities = [_positive_integer("cardinality", n) for n in cardinalities]
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if kind == "ple_visit_transformer" and self.visit_width % 4:
            raise ValueError("visit_width must be divisible by four attention heads")
        self.use_ple = kind in {"ple_resnet", "ple_visit_transformer"}
        self.use_skip = kind != "mlp_control"
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, self.embedding_dim) for n in self.cardinalities
        ])
        if self.use_ple:
            self.ple_weight = nn.Parameter(torch.empty(
                self.n_numeric, self.bins, self.embedding_dim))
            self.ple_bias = nn.Parameter(torch.zeros(self.n_numeric, self.embedding_dim))
            nn.init.normal_(self.ple_weight, std=1 / math.sqrt(self.bins))
            numeric_width = self.n_numeric * (self.embedding_dim + 1)
        else:
            numeric_width = 2 * self.n_numeric
        input_width = numeric_width + len(self.cardinalities) * self.embedding_dim
        self.input_projection = nn.Linear(input_width, self.width)
        self.blocks = nn.ModuleList([
            ResidualBlock(self.width, dropout) for _ in range(blocks)
        ])
        if kind == "ple_visit_transformer":
            # Instantiated after the common PLE tower so its seeded initial
            # tabular parameters match a standalone ple_resnet.
            self.code_embedding = nn.Embedding(44, 16)
            self.blank_code_embedding = nn.Parameter(torch.empty(16))
            nn.init.normal_(self.blank_code_embedding)
            self.outcome_embedding = nn.Embedding(4, 8)
            self.type_embedding = nn.Embedding(8, 8)
            self.aux_projection = nn.Linear(4, 16)
            self.event_projection = nn.Linear(48, self.visit_width)
            self.position = nn.Parameter(torch.empty(1, self.length, self.visit_width))
            self.cls = nn.Parameter(torch.empty(1, 1, self.visit_width))
            nn.init.normal_(self.position, std=.02)
            nn.init.normal_(self.cls, std=.02)
            layer = nn.TransformerEncoderLayer(
                d_model=self.visit_width, nhead=4,
                dim_feedforward=2 * self.visit_width, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.visit_encoder = nn.TransformerEncoder(
                layer, num_layers=2, enable_nested_tensor=False)
            self.representation_dim = self.width + self.visit_width
        else:
            self.representation_dim = self.width
        self.head = nn.Sequential(
            nn.LayerNorm(self.representation_dim), nn.Linear(self.representation_dim, 1),
        )

    def _tabular_representation(self, numeric, categorical, basis):
        if numeric.ndim != 2 or numeric.shape[1] != 2 * self.n_numeric:
            raise ValueError("numeric must contain N scaled values followed by N missing flags")
        batch = numeric.shape[0]
        if categorical.ndim != 2 or categorical.shape != (batch, len(self.cardinalities)):
            raise ValueError("Categorical tensor shape does not match fitted cardinalities")
        if categorical.dtype not in {torch.int32, torch.int64}:
            raise ValueError("Categorical indices must be integer tensors")
        categorical_values = [
            embedding(categorical[:, j]) for j, embedding in enumerate(self.embeddings)
        ]
        if self.use_ple:
            if basis is None or basis.shape != (batch, self.n_numeric, self.bins):
                raise ValueError("basis must have shape (batch, N, bins)")
            encoded = torch.einsum("bnk,nke->bne", basis, self.ple_weight) + self.ple_bias
            values = torch.cat([encoded.flatten(1), numeric[:, self.n_numeric:]], dim=1)
        else:
            values = numeric
        hidden = self.input_projection(torch.cat([values, *categorical_values], dim=1))
        for block in self.blocks:
            hidden = block(hidden, use_skip=self.use_skip)
        return hidden

    def _visit_representation(self, events, valid, batch):
        if events is None or events.shape != (batch, self.length, 60):
            raise ValueError("events must have shape (batch, length, 60)")
        if valid is None or valid.dtype != torch.bool or valid.shape != (batch, self.length):
            raise ValueError("valid must be a boolean (batch, length) mask")
        # Multiplying padding by zero would retain NaN. Replace it before any
        # matrix multiplication, embedding lookup or attention computation.
        safe = torch.where(valid.unsqueeze(-1), events, torch.zeros_like(events))
        codes = safe[..., :44]
        counts = codes.sum(dim=-1, keepdim=True)
        code_values = codes @ self.code_embedding.weight / counts.clamp_min(1)
        code_values = torch.where(counts > 0, code_values,
                                  self.blank_code_embedding.view(1, 1, -1))
        outcomes = safe[..., 44:48]
        types = safe[..., 48:56]
        # Valid source visits are one-hot; a zero vector receives the existing
        # unknown category, rather than being silently interpreted as pass.
        outcome_index = torch.where(outcomes.sum(-1) > 0, outcomes.argmax(-1), 3)
        type_index = torch.where(types.sum(-1) > 0, types.argmax(-1), 7)
        values = torch.cat([
            code_values, self.outcome_embedding(outcome_index),
            self.type_embedding(type_index), self.aux_projection(safe[..., -4:]),
        ], dim=-1)
        visits = self.event_projection(values) + self.position
        tokens = torch.cat([self.cls.expand(batch, -1, -1), visits], dim=1)
        mask = torch.cat([
            torch.zeros(batch, 1, dtype=torch.bool, device=valid.device), ~valid,
        ], dim=1)
        # The unmasked CLS token makes even an entirely empty history valid.
        return self.visit_encoder(tokens, src_key_padding_mask=mask)[:, 0]

    def representation(self, numeric, categorical, basis, events, valid):
        """Return the penultimate vector: width, or width + visit_width."""
        hidden = self._tabular_representation(numeric, categorical, basis)
        if self.kind == "ple_visit_transformer":
            visits = self._visit_representation(events, valid, len(numeric))
            hidden = torch.cat([hidden, visits], dim=1)
        return hidden

    def forward(self, numeric, categorical, basis, events, valid):
        return self.head(self.representation(
            numeric, categorical, basis, events, valid)).squeeze(-1)
