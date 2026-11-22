"""Two source-grounded CatBoost TD hypotheses and one temporal selector.

Adapted from pytabkit CB_TD_CLASS, not an exact reproduction: the source's
ordinary early stopping is replaced by our predeclared coarse inner-AP grid.
The second recipe changes only L2. No outer scores enter this selector.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .next_round import QUARTERS


SOURCE = "https://github.com/dholzmueller/pytabkit/blob/main/pytabkit/models/sklearn/default_params.py#L87-L92"
TD_GRID = (100, 250, 500, 1000)
_ORIGINAL = dict(
    learning_rate=.08, l2_leaf_reg=1e-5, boosting_type="Plain",
    grow_policy="SymmetricTree", bootstrap_type="Bernoulli", subsample=.9,
    depth=7, random_strength=.8, one_hot_max_size=15,
    leaf_estimation_iterations=1, border_count=254, rsm=1.,
)
TD_RECIPES = {
    "td_original": dict(id="td_original", family="catboost", view="full",
                         cap=1000, params=deepcopy(_ORIGINAL)),
    "td_l2_1": dict(id="td_l2_1", family="catboost", view="full",
                     cap=1000, params=dict(_ORIGINAL, l2_leaf_reg=1.)),
}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _score(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a numeric scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0 or (maximum is not None and result > maximum):
        raise ValueError(f"Invalid {name}")
    return result


def _selected_point(record, cap, outer_quarter):
    if not isinstance(record, Mapping):
        raise ValueError("Each old inner record must be a mapping")
    expected = (pd.Period(outer_quarter, freq="Q") - 1).start_time.strftime("%Y-%m-%d")
    if record.get("inner_quarter") != expected:
        raise ValueError("The selected inner quarter must strictly precede its outer quarter")
    curve = record.get("curve")
    if not isinstance(curve, (list, tuple)) or not curve:
        raise ValueError("Each inner record needs a nonempty saved curve")
    points = []
    for row in curve:
        count = _positive_integer(row["count"], "curve count")
        if count > cap:
            raise ValueError("A curve count exceeds its frozen recipe cap")
        points.append(dict(count=count, ap=_score(row["ap"], "inner AP", 1),
                           logloss=_score(row["logloss"], "inner logloss")))
    counts = [point["count"] for point in points]
    if counts != sorted(set(counts)):
        raise ValueError("Inner curve counts must be unique and increasing")
    selected = max(points, key=lambda point: (point["ap"], -point["count"]))
    if _positive_integer(record["best_count"], "best_count") != selected["count"]:
        raise ValueError("Saved best_count is not the earliest maximum-AP inner point")
    return dict(selected, inner_quarter=expected)


def choose_td(records, recipes=TD_RECIPES):
    """Return one TD tuner, using only earlier inner records for every choice.

    Per outer quarter: highest own-previous-inner AP, then lower logloss,
    earlier count and lexical recipe ID. The full recipe uses equal-weight mean
    of the three OLD INNER best APs; ties use mean logloss then lexical ID.
    The full count is the median of that chosen recipe's three selected counts.
    Synthetic fixtures may lower recipe caps without changing their parameters.
    """
    if (not isinstance(recipes, Mapping) or set(recipes) != set(TD_RECIPES)
            or not isinstance(records, Mapping) or set(records) != set(recipes)):
        raise ValueError("Exactly the two frozen TD recipes and their records are required")
    selected = {}
    for name, recipe in recipes.items():
        if (recipe.get("id") != name or recipe.get("family") != "catboost"
                or recipe.get("view") != "full" or recipe.get("params") != TD_RECIPES[name]["params"]):
            raise ValueError("TD recipe identity or parameters differ from the frozen pair")
        cap = _positive_integer(recipe["cap"], "recipe cap")
        if cap > 1000:
            raise ValueError("TD recipe cap cannot exceed1000")
        if not isinstance(records[name], Mapping) or set(records[name]) != set(QUARTERS):
            raise ValueError("Exactly the three old development quarter records are required")
        selected[name] = {q: _selected_point(records[name][q], cap, q) for q in QUARTERS}
    outer = {}
    for q in QUARTERS:
        name = min(recipes, key=lambda name: (-selected[name][q]["ap"], selected[name][q]["logloss"],
                                               selected[name][q]["count"], name))
        point = selected[name][q]
        outer[q] = dict(recipe=name, count=point["count"], inner_quarter=point["inner_quarter"],
                        inner_ap=point["ap"])
    name = min(recipes, key=lambda name: (-float(np.mean([selected[name][q]["ap"] for q in QUARTERS])),
                                         float(np.mean([selected[name][q]["logloss"] for q in QUARTERS])), name))
    counts = [selected[name][q]["count"] for q in QUARTERS]
    final = dict(recipe=name, count=int(np.median(counts)), old_inner_counts=counts,
                 mean_old_inner_best_ap=float(np.mean([selected[name][q]["ap"] for q in QUARTERS])))
    return dict(outer=outer, final=final)
