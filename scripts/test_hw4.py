"""Run the isolated HW4 checks without requiring pytest or the HW3 tests."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]


def main():
    count = 0
    for name in ["test_hw4_features", "test_hw4_extensions", "test_hw4_neural", "test_hw4_masked",
                 "test_hw4_classical_features", "test_hw4_classical_models",
                 "test_hw4_classical_analysis", "test_hw4_classical_runner",
                 "test_hw4_residual_preprocessing", "test_hw4_residual_models",
                 "test_hw4_residual_training", "test_hw4_residual_runner",
                 "test_hw4_delivery", "test_hw4_delivery_neural",
                 "test_hw4_map_ranking", "test_hw4_next_round", "test_hw4_tabm_models",
                 "test_hw4_site_state", "test_hw4_tabm_training",
                 "test_hw4_deep_boosting", "test_hw4_transition_features",
                 "test_hw4_seasonal_context", "test_hw4_final_research",
                 "test_hw4_additive_logit", "test_hw4_additive_runner",
                 "test_hw4_context_bag10", "test_hw4_inner_ap_followup"]:
        module = importlib.import_module(name)
        for key, value in vars(module).items():
            if key.startswith("test_") and callable(value):
                value()
                count += 1
                print(f"{name}.{key}: PASS", flush=True)
    print(f"{count} HW4 checks passed")


if __name__ == "__main__":
    main()
