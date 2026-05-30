"""Run the full reproduction pipeline, in order.

This executes every analysis script as a separate process (so each runs with a
clean state) and stops if any of them fails. After it finishes, every table in
tables/ and every figure in figures/ will have been regenerated from the data
in data/. The figure step runs last because it draws from the tables the
earlier steps produce.

    python3 run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent

STEPS = [
    "01_summary_statistics.py",
    "02_production_function.py",
    "03_factor_shares.py",
    "04_dual_channel.py",
    "05_apex_friction.py",
    "06_measurement_calibration.py",
    "07_make_figures.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n{'=' * 70}\nRUNNING {step}\n{'=' * 70}")
        result = subprocess.run([sys.executable, str(CODE / step)], cwd=CODE)
        if result.returncode != 0:
            raise SystemExit(f"{step} failed with exit code "
                             f"{result.returncode}; pipeline stopped.")
    print(f"\n{'=' * 70}\nALL STEPS COMPLETE\n{'=' * 70}")
    print("Tables are in ../tables/ and figures are in ../figures/.")


if __name__ == "__main__":
    main()
