"""Synthetic-only checks for matched scheduler and canonical EBM policies."""
from copy import deepcopy
from io import BytesIO

import joblib
import numpy as np

from hw4_inspections.ebm_search import EBM_RECIPES, FIXED_PAIRS
from hw4_inspections.ebm_canonical import CANONICAL_RECIPES, fit_canonical, predict_canonical
from test_hw4_ebm_search import RUNTIME, fixture, expect_error


def tiny(name):
    recipe = deepcopy(CANONICAL_RECIPES[name])
    recipe["params"]["max_rounds"] = 2 if name.startswith("ebm_match_") else 5
    if name.startswith("ebm_canonical_"):
        recipe["params"]["outer_bags"] = 2
    return recipe


def test_canonical_four_exact_source_scopes_and_matched_only_scheduler_change():
    assert len(CANONICAL_RECIPES) == 4
    for terms in ["main", "pairs10"]:
        matched = CANONICAL_RECIPES[f"ebm_match_{terms}_300"]["params"]
        original = EBM_RECIPES[f"ebm_compact94_{terms}_300"]["params"]
        assert {k for k in matched if matched[k] != original[k]} == {"greedy_ratio"}
        canonical = CANONICAL_RECIPES[f"ebm_canonical_{terms}"]["params"]
        assert canonical["learning_rate"] == .015 and canonical["max_bins"] == 1024
        assert canonical["max_interaction_bins"] == 64 and canonical["min_samples_leaf"] == 4
        assert canonical["reg_lambda"] == 0 and canonical["max_leaves"] == 2
        assert canonical["max_rounds"] == 2000 and canonical["outer_bags"] == 8
        assert canonical["validation_size"] == .15 and canonical["early_stopping_rounds"] == 50
        assert canonical["early_stopping_tolerance"] == 1e-5
        assert canonical["greedy_ratio"] == 10 and canonical["cyclic_progress"] is False
        assert canonical["interactions"] == 0 or [tuple(p) for p in canonical["interactions"]] == list(FIXED_PAIRS)


def test_canonical_four_native_tiny_fits_reload_metadata_missing_and_term_scope():
    x, y, cats, valid = fixture()
    for name in CANONICAL_RECIPES:
        bundle = fit_canonical(tiny(name), x, y, cats, runtime=RUNTIME, threads=2)
        p = predict_canonical(bundle, valid)
        assert p.dtype == np.float64 and p.shape == (4,) and np.isfinite(p).all()
        assert len(bundle["terms"]) == 94 + (10 if "pairs10" in name else 0)
        assert bundle["effective_params"]["greedy_ratio"] == 10
        assert bundle["effective_params"]["cyclic_progress"] is False
        assert "NOT forced" in bundle["cyclic_note"]
        updates = np.asarray(bundle["stage_update_counts"])
        assert updates.shape == (2 if "pairs10" in name else 1,
                                 2 if name.startswith("ebm_canonical_") else 1)
        assert (updates > 0).all()
        assert bundle["outer_labels_seen"] is False
        assert ("random15%" in bundle["internal_validation_policy"]) == name.startswith("ebm_canonical_")
        stream = BytesIO()
        joblib.dump(bundle, stream)
        stream.seek(0)
        np.testing.assert_array_equal(p, predict_canonical(joblib.load(stream), valid))
        np.testing.assert_array_equal(p[::-1], predict_canonical(bundle, valid.iloc[::-1]))
        np.testing.assert_array_equal(p[:1], p[1:2])


def test_canonical_threads_scope_and_internal_validation_guards():
    x, y, cats, _ = fixture()
    for threads in [0, 3, -1, True]:
        expect_error(lambda threads=threads: fit_canonical(tiny("ebm_canonical_main"), x, y, cats,
                                                          runtime=RUNTIME, threads=threads))
    invalid = tiny("ebm_match_main_300")
    invalid["params"]["validation_size"] = .15
    expect_error(lambda: fit_canonical(invalid, x, y, cats, runtime=RUNTIME))
    invalid = tiny("ebm_canonical_main")
    invalid["params"]["outer_bags"] = 9
    expect_error(lambda: fit_canonical(invalid, x, y, cats, runtime=RUNTIME))
    invalid = tiny("ebm_canonical_pairs10")
    invalid["params"]["interactions"] = 10
    expect_error(lambda: fit_canonical(invalid, x, y, cats, runtime=RUNTIME))
    invalid = tiny("ebm_canonical_main")
    invalid["params"]["max_rounds"] = 2001
    expect_error(lambda: fit_canonical(invalid, x, y, cats, runtime=RUNTIME))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS", flush=True)
