"""Matched default-greedy and canonical-style EBM continuation.

Only supplied past fit rows enter the native estimator. Canonical-style bags
use random validation inside that past fit pool, per the explicitly approved
upstream procedure; no external inner or outer evaluation frame is accepted.
This is not the full upstream default: bag count, cap and fixed pair scope are
bounded resource adaptations recorded with each model.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import warnings

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .ebm_search import (
    EBM_RECIPES, FIXED_PAIRS, VERSION, _categorical_values, _integer, _transform,
    load_runtime, predict_ebm, select_columns,
)


def _recipes():
    result = {}
    for terms in ("main", "pairs10"):
        source = EBM_RECIPES[f"ebm_compact94_{terms}_300"]
        matched = deepcopy(source["params"])
        matched["greedy_ratio"] = 10.
        name = f"ebm_match_{terms}_300"
        result[name] = dict(id=name, view="compact94", params=matched)
        canonical = dict(matched, learning_rate=.015, max_bins=1024,
                         max_interaction_bins=64, min_samples_leaf=4, reg_lambda=0.,
                         validation_size=.15, outer_bags=8, inner_bags=0,
                         max_rounds=2000, early_stopping_rounds=50,
                         early_stopping_tolerance=1e-5)
        name = f"ebm_canonical_{terms}"
        result[name] = dict(id=name, view="compact94", params=canonical)
    return result


CANONICAL_RECIPES = _recipes()


def fit_canonical(recipe, x, y, cats, *, runtime, seed=42, threads=2):
    """Fit one approved EBM recipe, with optional internal past-only bags."""
    seed = _integer(seed, "seed", minimum=0)
    threads = _integer(threads, "threads")
    if threads > 2:
        raise ValueError("Canonical EBM has a hard maximum of two CPU jobs")
    if recipe.get("id") not in CANONICAL_RECIPES or recipe.get("view") != "compact94":
        raise ValueError("Only the four frozen compact94 EBM recipes are allowed")
    target = load_runtime(runtime)
    from interpret.glassbox import ExplainableBoostingClassifier
    from interpret.utils._native import Native

    params = deepcopy(recipe["params"])
    params["max_rounds"] = _integer(params["max_rounds"], "max_rounds")
    if params["max_rounds"] > 2000:
        raise ValueError("The bounded EBM cap is2000 cycles")
    if params.get("greedy_ratio") != 10 or params.get("cyclic_progress") is not False:
        raise ValueError("This continuation tests the fixed upstream greedy10 scheduler")
    canonical = recipe["id"].startswith("ebm_canonical_")
    if canonical:
        bags = _integer(params["outer_bags"], "outer_bags")
        if (bags > 8 or params["validation_size"] != .15 or params["early_stopping_rounds"] != 50
                or params["early_stopping_tolerance"] != 1e-5 or params["inner_bags"] != 0):
            raise ValueError("Canonical-style validation/bagging policy differs from the approved recipe")
    elif (params.get("validation_size") != 0 or params.get("early_stopping_rounds") != 0
          or params.get("outer_bags") != 1 or params.get("inner_bags") != 0):
        raise ValueError("Matched greedy recipes retain the original no-validation policy")
    columns = select_columns(x, "compact94")
    cats = list(cats)
    if len(cats) != len(set(cats)) or not set(cats).issubset(x.columns):
        raise ValueError("Categories must be unique input column names")
    categorical = [name for name in columns if name in cats or name == "forecast_quarter"]
    labels = np.asarray(y)
    if labels.shape != (len(x),) or not np.isin(labels, [0, 1]).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("EBM fit needs aligned binary labels and both classes")
    bundle = dict(recipe=deepcopy(recipe), view="compact94", input_features=list(x),
                  features=columns, categorical=categorical,
                  vocabularies={name: sorted(_categorical_values(x, name).dropna().unique().tolist())
                                for name in categorical},
                  runtime_target=str(target), library_version=VERSION, seed=seed, threads=threads)
    prepared = _transform(bundle, x)
    named_pairs = params["interactions"]
    if named_pairs == 0:
        pairs = []
        params["interactions"] = 0
    else:
        if (not isinstance(named_pairs, (list, tuple))
                or [tuple(pair) for pair in named_pairs] != list(FIXED_PAIRS)):
            raise ValueError("Only the ten fixed named pairs are permitted")
        pairs = [(columns.index(a), columns.index(b)) for a, b in FIXED_PAIRS]
        params["interactions"] = pairs
    types = ["nominal" if name in categorical else "continuous" for name in columns]
    params.update(feature_names=columns, feature_types=types, random_state=seed, n_jobs=threads)
    model = ExplainableBoostingClassifier(**params)
    with threadpool_limits(limits=threads), joblib.parallel_config(backend="loky", inner_max_num_threads=1), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module=r"interpret\.glassbox\._ebm\._ebm",
            message=r"Missing values detected\. Our visualizations do not currently display missing values\..*",
        )
        model.fit(prepared, labels.astype(np.int64))
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected native EBM binary class order")
    actual = [tuple(term) for term in model.term_features_]
    expected = {(i,) for i in range(len(columns))} | {tuple(sorted(pair)) for pair in pairs}
    if len(actual) != len(expected) or {tuple(sorted(term)) for term in actual} != expected:
        raise ValueError("Native EBM selected unplanned terms")
    native = Path(Native._get_ebm_lib_path())
    with native.open("rb") as stream:
        native_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    validation = (
        "Internal random15% validation per outer bag ONLY inside the caller-supplied PAST fit rows; bins see this full past fit pool per upstream. External inner/outer evaluation rows and labels were not supplied."
        if canonical else
        "No internal validation or early stopping; all caller-supplied PAST fit rows form one native bag."
    )
    bundle.update(model=model, effective_params=model.get_params(deep=False),
                  native_library=str(native), native_library_sha256=native_sha,
                  feature_types=types, raw_known_feature_types=dict(zip(columns, types)),
                  terms=[list(term) for term in actual], interaction_indices=pairs,
                  interaction_names=[] if not pairs else list(FIXED_PAIRS),
                  max_rounds=params["max_rounds"], trained_rows=len(x), outer_labels_seen=False,
                  stage_update_counts=np.asarray(model.best_iteration_, dtype=np.int64).tolist(),
                  update_count_meaning="Native best_iteration_[stage,bag] is attempted update steps before stopping, not a retained-round prefix",
                  internal_validation_policy=validation,
                  cyclic_note="Public greedy_ratio10 and cyclic_progressFalse; after smoothing cyclic passes refresh training gains, greedy passes apply updates. Cyclic progress is NOT forced to1.",
                  adaptation="Canonical-style uses bounded2000cap/8bags/ES50 and zero or ten fixed interactions, not every upstream default")
    return bundle


predict_canonical = predict_ebm
