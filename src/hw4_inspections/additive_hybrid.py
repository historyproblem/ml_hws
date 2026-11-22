"""Twelve frozen interaction-GAM x recency/L2 combinations; inner selection."""
from __future__ import annotations

from collections import Counter
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .additive_search import CLASSIFIER, SearchTransform, predict_recipe
from .classical_features import recency_weights


RECIPES = tuple(dict(id=f"hybrid_H{h}_C{c:g}", half_life_quarters=h, C=c)
                for h in (2, 4, 8) for c in (.003, .01, .03, .1))


def recipe_by_id(name):
    for recipe in RECIPES:
        if recipe["id"] == name:
            return dict(recipe)
    raise ValueError(f"Unknown frozen hybrid recipe {name}")


def fit_recipe(frame, y, recipe, *, periods, reference):
    recipe = recipe_by_id(recipe["id"])
    target = np.asarray(y)
    if target.shape != (len(frame),) or not np.isfinite(target).all() or set(np.unique(target)) != {0, 1}:
        raise ValueError("Aligned binary labels with both classes required")
    if len(periods) != len(frame):
        raise ValueError("Training periods must align with all fit rows")
    weights = recency_weights(periods, reference, half_life_quarters=recipe["half_life_quarters"])
    total = float(weights.sum())
    actual_l2 = 1. / (recipe["C"] * total)
    unweighted_scale_l2 = 1. / (recipe["C"] * len(frame))
    if not np.isclose(weights.mean(), 1., atol=1e-12, rtol=1e-12):
        raise AssertionError("Recency normalization changed global C scale")
    transform = SearchTransform("core_interactions").fit(frame)
    model = LogisticRegression(C=recipe["C"], **CLASSIFIER)
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        model.fit(transform.transform(frame), target, sample_weight=weights)
    return dict(transformer=transform, model=model, recipe=recipe,
        classifier={"C":recipe["C"],**CLASSIFIER}, iterations=int(model.n_iter_.max()),
        feature_names=transform.feature_names_, threads=2,
        weight_summary=dict(mean=float(weights.mean()), sum=total, minimum=float(weights.min()),
            maximum=float(weights.max()), effective_sample_size=total**2 / float(np.square(weights).sum()),
            sample_rows=len(frame), l2_strength_actual=actual_l2,
            l2_strength_if_unit_weights=unweighted_scale_l2,
            l2_scale_ratio=actual_l2/unweighted_scale_l2),
        sample_weights=f"mean-one calendar-quarter half-life{recipe['half_life_quarters']}")


def choose_inner(records):
    valid=[r for r in records if r["status"]=="complete" and np.isfinite(r["ap"])]
    if not valid:
        raise RuntimeError("All hybrid recipes invalid; cannot omit an inner/outer quarter")
    return min(valid,key=lambda r:(-r["ap"],recipe_by_id(r["recipe"])["C"],
        -recipe_by_id(r["recipe"])["half_life_quarters"],r["recipe"]))["recipe"]


def choose_final(winners,records):
    votes=Counter(winners);valid={}
    for recipe in RECIPES:
        rows=[r for r in records if r["recipe"]==recipe["id"]]
        if len(rows)==3 and all(r["status"]=="complete" for r in rows):
            valid[recipe["id"]]=float(np.mean([r["ap"] for r in rows]))
    if not valid:
        raise RuntimeError("No hybrid recipe converged in all3inner quarters")
    return min(valid,key=lambda name:(-votes[name],-valid[name],recipe_by_id(name)["C"],
        -recipe_by_id(name)["half_life_quarters"],name))
