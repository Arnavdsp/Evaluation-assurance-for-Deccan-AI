"""Runs every experiment, then regenerates figures. ~2 minutes on a laptop."""
import runpy
from pathlib import Path
HERE = Path(__file__).resolve().parent
for s in ["headline", "realistic_prevalence", "per_mode", "calibration_negative_control",
          "robustness_shift", "canary_size", "make_figures"]:
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")
    runpy.run_path(str(HERE / f"{s}.py"), run_name="__main__")
