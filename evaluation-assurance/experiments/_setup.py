"""Makes `src/` importable from any working directory and fixes output paths."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
FIGURES = ROOT / "figures"; FIGURES.mkdir(exist_ok=True)
N_SEEDS = 20
BUDGETS = [0.05, 0.10, 0.20, 0.30]
