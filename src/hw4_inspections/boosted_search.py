"""Frozen CPU boosting paths for nested, past-only configuration selection.

The caller owns temporal splits. This module sees only the supplied fit and
inner-evaluation frames. LightGBM checkpoints are serialized when their round
is reached: this matters for DART, which later reweights earlier trees.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from numbers import Integral
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
from threadpoolctl import threadpool_limits


CAP = 1600
GRID = (100, 250, 500, 1000, 1600)
FORBIDDEN = frozenset({"target", "row_id", "site_id", "forecast_period", "history_before", "inspection_week"})
_GBDT = (
    (7, 3, 20, 1., .8, False),
    (15, 4, 25, 1., .8, False),
    (31, 6, 30, 1., .8, False),
    (63, 8, 40, 3., .8, False),
    (15, 5, 50, 0., .6, False),
    (31, 6, 60, 5., 1., False),
    (31, 6, 20, 1., .5, True),
    (63, 8, 80, 10., .7, False),
)
_DART = ((15, 5, 40, 3.), (31, 6, 60, 5.))


def _recipes():
    recipes = {}
    common = dict(learning_rate=.03, bagging_fraction=.8, bagging_freq=1,
                  cat_l2=3., cat_smooth=10., max_bin=255)
    for view in ("full", "no_codes"):
        for i, (leaves, depth, leaf, l2, fraction, extra) in enumerate(_GBDT, 1):
            name = f"gbdt_{view}_{i:02d}"
            recipes[name] = dict(id=name, family="lgb_gbdt", view=view, cap=CAP, params=dict(
                **common, num_leaves=leaves, max_depth=depth, min_data_in_leaf=leaf,
                lambda_l2=l2, feature_fraction=fraction, extra_trees=extra))
        for i, (leaves, depth, leaf, l2) in enumerate(_DART, 1):
            name = f"dart_{view}_{i:02d}"
            recipes[name] = dict(id=name, family="lgb_dart", view=view, cap=CAP, params=dict(
                **common, num_leaves=leaves, max_depth=depth, min_data_in_leaf=leaf,
                lambda_l2=l2, feature_fraction=.8, drop_rate=.1, skip_drop=.5,
                max_drop=50, uniform_drop=False))
    for bootstrap, depth, l2 in (("Bernoulli", 5, 3.), ("Bernoulli", 6, 10.),
                                  ("Bayesian", 4, 3.), ("Bayesian", 6, 10.)):
        name = f"cb_{bootstrap.lower()}_d{depth}_l2_{int(l2)}"
        params = dict(boosting_type="Plain", grow_policy="SymmetricTree", depth=depth,
                      l2_leaf_reg=l2, learning_rate=.03, rsm=.7, border_count=254,
                      bootstrap_type=bootstrap)
        params.update(subsample=.8) if bootstrap == "Bernoulli" else params.update(bagging_temperature=.8)
        recipes[name] = dict(id=name, family="catboost", view="full", params=params, cap=CAP)
    return recipes


RECIPES = _recipes()


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _threads(value):
    value = _integer(value, "threads")
    if value > 2:
        raise ValueError("This search has a hard maximum of two CPU threads per model")
    return value


def _check_frame(frame):
    if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
            or not all(isinstance(c, str) for c in frame) or not len(frame.columns)):
        raise ValueError("Features must have nonempty unique string columns")
    if FORBIDDEN.intersection(frame.columns):
        raise ValueError("Current target and raw row/site/date identifiers are forbidden")
    if any(pd.api.types.is_datetime64_any_dtype(frame[c]) for c in frame):
        raise ValueError("Raw datetime values cannot be input features")


def select_columns(frame, view):
    """Return the frozen full or no-code view in the supplied column order."""
    _check_frame(frame)
    if view not in {"full", "no_codes"}:
        raise ValueError("Unknown feature view")
    columns = [c for c in frame if view == "full" or not c.startswith("code")]
    if not columns:
        raise ValueError("The selected feature view is empty")
    return columns


def _labels(y, size, name):
    target = np.asarray(y)
    if target.shape != (size,) or not np.isin(target, [0, 1]).all() or set(np.unique(target)) != {0, 1}:
        raise ValueError(f"{name} needs aligned binary targets and both classes")
    return target.astype(np.int64, copy=True)


def _initial(recipe, x, cats, seed, threads):
    if recipe["family"] not in {"lgb_gbdt", "lgb_dart", "catboost"}:
        raise ValueError("Unknown boosting family")
    seed = _integer(seed, "seed", minimum=0)
    threads = _threads(threads)
    columns = select_columns(x, recipe["view"])
    cats = list(cats)
    if len(cats) != len(set(cats)) or not set(cats).issubset(x.columns):
        raise ValueError("Categories must be unique feature columns")
    selected_cats = [name for name in cats if name in columns]
    bundle = dict(recipe=deepcopy(recipe), family=recipe["family"], view=recipe["view"],
                  input_features=list(x), features=columns, cats=selected_cats,
                  seed=seed, threads=threads, preprocessor=None)
    if recipe["family"].startswith("lgb_"):
        bundle["preprocessor"] = {name: sorted(x[name].astype("string").dropna().unique().tolist())
                                  for name in selected_cats}
        bundle["all_missing_categorical_columns"] = [name for name, values in bundle["preprocessor"].items() if not values]
    return bundle, _transform(bundle, x)


def _transform(bundle, x):
    _check_frame(x)
    if list(x) not in [bundle["input_features"], bundle["features"]]:
        raise ValueError("Prediction feature schema/order differs from fit")
    prepared = x[bundle["features"]].copy(deep=True)
    numeric = [name for name in prepared if name not in bundle["cats"]]
    prepared[numeric] = prepared[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    for name in bundle["cats"]:
        if bundle["family"] == "catboost":
            prepared[name] = prepared[name].fillna("<none>").astype(str)
        else:
            vocabulary = bundle["preprocessor"][name]
            prepared[name] = (pd.Categorical(prepared[name].astype("string"), categories=vocabulary)
                              if vocabulary else np.nan)
    return prepared


def _lgb_params(recipe, seed, threads):
    return dict(recipe["params"], boosting_type="dart" if recipe["family"] == "lgb_dart" else "gbdt",
                objective="binary", metric="None", verbosity=-1, device_type="cpu", num_threads=threads,
                deterministic=True, force_col_wise=True, feature_pre_filter=False,
                use_missing=True, zero_as_missing=False, seed=seed, data_random_seed=seed,
                feature_fraction_seed=seed, bagging_seed=seed, drop_seed=seed)


def _parameter_text(model_string):
    start = model_string.find("\nparameters:\n")
    end = model_string.find("\nend of parameters", start)
    return model_string[start + 1:end] if start >= 0 and end >= 0 else None


def _lgb_bundle(bundle, model, count, trained_cap, params):
    import lightgbm

    bundle.update(model=model, count=count, trained_cap=trained_cap,
                  actual_tree_count=int(model.current_iteration()), library_version=lightgbm.__version__,
                  effective_params=dict(params), selected_count=count)
    text = model.model_to_string(num_iteration=-1)
    bundle["serialized_parameters"] = _parameter_text(text)
    bundle["model_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return bundle


def _cat_model(recipe, count, seed, threads):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(iterations=count, loss_function="Logloss", task_type="CPU",
                             random_seed=seed, thread_count=threads, allow_writing_files=False,
                             verbose=False, **recipe["params"])


def _cat_bundle(bundle, model, count, trained_cap):
    import catboost

    bundle.update(model=model, count=count, trained_cap=trained_cap, selected_count=count,
                  actual_tree_count=int(model.tree_count_), effective_params=model.get_all_params(),
                  library_version=catboost.__version__)
    return bundle


def _probabilities(values, size):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (size,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Predictions must be aligned finite class1 probabilities")
    return values


def fit_fixed(recipe, x, y, cats, *, count, seed=42, threads=2):
    """Fit one fresh outer/full model at an already selected fixed round count."""
    count = _integer(count, "count")
    if count > _integer(recipe["cap"], "recipe cap"):
        raise ValueError("Count exceeds the frozen recipe cap")
    bundle, prepared = _initial(recipe, x, cats, seed, threads)
    target = _labels(y, len(x), "Fit")
    with threadpool_limits(limits=bundle["threads"]):
        if recipe["family"] == "catboost":
            model = _cat_model(recipe, count, bundle["seed"], bundle["threads"])
            model.fit(prepared, target, cat_features=bundle["cats"])
            return _cat_bundle(bundle, model, count, count)
        import lightgbm as lgb

        params = _lgb_params(recipe, bundle["seed"], bundle["threads"])
        dataset = lgb.Dataset(prepared, label=target, free_raw_data=False,
                              categorical_feature=[name for name in bundle["cats"]
                                                   if name not in bundle["all_missing_categorical_columns"]])
        model = lgb.train(params, dataset, num_boost_round=count)
        return _lgb_bundle(bundle, model, count, count, params)


def predict(bundle, x):
    """Use the saved category vocabulary and actual retained checkpoint."""
    prepared = _transform(bundle, x)
    if not len(x):
        return np.empty(0, dtype=np.float64)
    with threadpool_limits(limits=bundle["threads"]):
        if bundle["family"] == "catboost":
            values = bundle["model"].predict_proba(prepared, thread_count=bundle["threads"])[:, 1]
        else:
            values = bundle["model"].predict(prepared, num_iteration=-1,
                                               num_threads=bundle["threads"], validate_features=True)
    return _probabilities(values, len(x))


def fit_path(recipe, xfit, yfit, xeval, yeval, cats, *, seed=42, threads=2, cap=None, grid=None):
    """One max-size inner fit with true in-time checkpoints and AP selection.

    With grid=None the frozen grid is filtered by cap and cap is included. An
    explicit grid is honored exactly, including when it omits the final cap.
    Equal AP chooses the earlier count. Logloss is a report, not a selector.
    """
    start = time.monotonic()
    maximum = _integer(recipe["cap"] if cap is None else cap, "cap")
    if maximum > _integer(recipe["cap"], "recipe cap"):
        raise ValueError("Inner cap exceeds the frozen recipe")
    counts = sorted({n for n in GRID if n <= maximum} | {maximum}) if grid is None else list(grid)
    if (not counts or any(_integer(n, "grid count") > maximum for n in counts)
            or counts != sorted(set(counts))):
        raise ValueError("Grid must contain strictly increasing positive counts no larger than cap")
    bundle, fit = _initial(recipe, xfit, cats, seed, threads)
    target = _labels(yfit, len(xfit), "Fit")
    evaluation = _transform(bundle, xeval)
    eval_target = _labels(yeval, len(xeval), "Inner evaluation")
    curve, best = [], None

    def record(count, values, snapshot):
        nonlocal best
        probability = _probabilities(values, len(xeval))
        row = dict(count=int(count), ap=float(average_precision_score(eval_target, probability)),
                   logloss=float(log_loss(eval_target, probability, labels=[0, 1])))
        curve.append(row)
        if best is None or row["ap"] > best["ap"]:
            best = dict(row, snapshot=snapshot())

    with threadpool_limits(limits=bundle["threads"]):
        if recipe["family"] == "catboost":
            model = _cat_model(recipe, maximum, bundle["seed"], bundle["threads"])
            model.fit(fit, target, cat_features=bundle["cats"])
            for count in counts:
                probability = model.predict_proba(evaluation, ntree_end=count,
                                                    thread_count=bundle["threads"])[:, 1]
                # CB trees are additive after fitting; cloning and shrinking a
                # retained prefix keeps the selected model independent.
                def checkpoint(count=count):
                    result = model.copy()
                    result.shrink(count)
                    return result
                record(count, probability, checkpoint)
            selected = _cat_bundle(bundle, best["snapshot"], best["count"], maximum)
        else:
            import lightgbm as lgb

            params = _lgb_params(recipe, bundle["seed"], bundle["threads"])
            dataset = lgb.Dataset(fit, label=target, free_raw_data=False,
                                  categorical_feature=[name for name in bundle["cats"]
                                                       if name not in bundle["all_missing_categorical_columns"]])

            def checkpoint_callback(env):
                count = env.iteration + 1
                if count in counts:
                    # DART may later rescale earlier trees. Evaluate and save
                    # the model NOW, never truncate the final maximum booster.
                    value = env.model.predict(evaluation, num_iteration=-1,
                                              num_threads=bundle["threads"])
                    record(count, value, lambda: env.model.model_to_string(num_iteration=-1))

            checkpoint_callback.before_iteration = False
            checkpoint_callback.order = 50
            lgb.train(params, dataset, num_boost_round=maximum, callbacks=[checkpoint_callback])
            if best is None or [row["count"] for row in curve] != counts:
                raise AssertionError("The actual checkpoint grid differs from its frozen definition")
            model = lgb.Booster(model_str=best["snapshot"])
            selected = _lgb_bundle(bundle, model, best["count"], maximum, params)
    selected["checkpoint_selection"] = "inner sklearnAP maximum; earliest count tie; genuine checkpoint"
    if not np.array_equal(predict(selected, xeval), _probabilities(
            selected["model"].predict_proba(evaluation, thread_count=bundle["threads"])[:, 1]
            if recipe["family"] == "catboost" else selected["model"].predict(
                evaluation, num_iteration=-1, num_threads=bundle["threads"]), len(xeval))):
        raise AssertionError("Retained model and saved preprocessing give different predictions")
    return dict(best_count=best["count"], curve=curve, bundle=selected, seconds=time.monotonic() - start)
