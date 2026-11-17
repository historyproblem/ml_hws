"""Run the isolated HW4 checks without requiring pytest or the HW3 tests."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]


def main():
    count = 0
    for name in ["test_hw4_features", "test_hw4_extensions", "test_hw4_neural", "test_hw4_masked"]:
        module = importlib.import_module(name)
        for key, value in vars(module).items():
            if key.startswith("test_") and callable(value):
                value()
                count += 1
                print(f"{name}.{key}: PASS", flush=True)
    print(f"{count} HW4 checks passed")


if __name__ == "__main__":
    main()
