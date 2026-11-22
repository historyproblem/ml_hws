"""Run the isolated HW4 checks without requiring pytest or the HW3 tests."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true", help="Include the completed six-hour CPU and archive checks")
    parser.add_argument("--ebm-runtime", type=Path, help="Optional isolated interpret-core0.7.8 site directory")
    parser.add_argument("--start-at", help="Resume an interrupted check invocation at a named module")
    args = parser.parse_args()
    if args.ebm_runtime:
        os.environ["HW4_EBM_RUNTIME"] = str(args.ebm_runtime.resolve())
    count = 0
    modules = ["test_hw4_features", "test_hw4_extensions", "test_hw4_neural", "test_hw4_masked",
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
                 "test_hw4_context_bag10", "test_hw4_inner_ap_followup"]
    if args.extended:
        modules += [
            "test_hw4_additive_search", "test_hw4_additive_search_runner",
            "test_hw4_additive_hybrid", "test_hw4_additive_hybrid_runner",
            "test_hw4_boosted_search", "test_hw4_boosted_runner",
            "test_hw4_event_learning", "test_hw4_event_library",
            "test_hw4_gam_residual", "test_hw4_gam_residual_runner",
            "test_hw4_stacking", "test_hw4_stacking_runner",
            "test_hw4_gam_state", "test_hw4_gam_state_runner",
            "test_hw4_td_presets", "test_hw4_td_runner",
            "test_hw4_rbf_search", "test_hw4_rbf_runner",
            "test_hw4_rbf_convergence", "test_hw4_rbf_convergence_runner",
            "test_hw4_tabm_preprocessing", "test_hw4_tabm_preprocessing_runner",
            "test_hw4_tabm_preprocessing_v100", "test_hw4_extended_v100",
            "test_hw4_tabpfn_v100", "test_hw4_tabicl_v100",
            "test_hw4_safe_archives", "test_hw4_gpu_unpack", "test_hw4_foundation_consumers",
            "test_hw4_recent_finalists", "test_hw4_recent_baselines",
        ]
    if args.ebm_runtime:
        modules += ["test_hw4_ebm_search", "test_hw4_ebm_runner",
                    "test_hw4_ebm_canonical", "test_hw4_ebm_canonical_runner", "test_hw4_recent_ebm"]
    if args.start_at:
        if args.start_at not in modules:
            raise ValueError("Requested start module is not in the selected suite")
        modules = modules[modules.index(args.start_at):]
    for name in modules:
        module = importlib.import_module(name)
        for key, value in list(vars(module).items()):
            if key.startswith("test_") and callable(value):
                value()
                count += 1
                print(f"{name}.{key}: PASS", flush=True)
    print(f"{count} HW4 checks passed")


if __name__ == "__main__":
    main()
