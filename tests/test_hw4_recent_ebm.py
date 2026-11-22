"""A native tiny past-only fit exercises unscored recent EBM persistence."""
from argparse import Namespace
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_hw4_recent_ebm as runner
from hw4_inspections.ebm_canonical import CANONICAL_RECIPES
from test_hw4_additive_runner import fixture
from test_hw4_ebm_search import RUNTIME


def test_recent_ebm_native_past_only_no_external_eval_exact_resume():
    with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
        args = fixture(Path(temporary))
        args.runtime = RUNTIME
        cache = joblib.load(args.feature_cache)
        train, frame, cats = cache["train"], cache["views"]["base"], cache["cats"]
        recipe = deepcopy(CANONICAL_RECIPES[runner.RECIPE])
        recipe["params"]["max_rounds"] = 3
        recipe["params"]["outer_bags"] = 1
        original = runner.fit_canonical
        calls = []
        def checked(spec, x, y, categories, **kwargs):
            assert train.forecast_period.iloc[x.index].lt(runner.QUARTERS[0]).all()
            assert "eval_frame" not in kwargs and "eval_target" not in kwargs
            assert kwargs["threads"] == 2
            calls.append(len(x))
            return original(spec, x, y, categories, **kwargs)
        with patch.object(runner, "fit_canonical", side_effect=checked):
            value, marker = runner.job(args, "synthetic", runner.QUARTERS[0], 42, recipe, train, frame, cats)
        assert len(calls) == 1 and value.shape == (16,)
        assert marker["reload_exact"] and marker["metrics_computed"] == 0 and not marker["outer_labels_seen"]
        args.resume = True
        with patch.object(runner, "fit_canonical", side_effect=AssertionError("Unexpected resume fit")):
            replay, same = runner.job(args, "synthetic", runner.QUARTERS[0], 42, recipe, train, frame, cats)
        assert np.array_equal(value, replay) and same == marker
        payload = args.out / "models" / f"{runner.QUARTERS[0]}__s42" / "prediction.npy"
        payload.write_bytes(payload.read_bytes() + b"corrupt")
        try:
            runner.job(args, "synthetic", runner.QUARTERS[0], 42, recipe, train, frame, cats)
            raise AssertionError("Corrupted recent EBM checkpoint accepted")
        except ValueError:
            pass
