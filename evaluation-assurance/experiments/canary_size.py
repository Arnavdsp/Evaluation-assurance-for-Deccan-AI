"""How many human labels does the auditor need? Sweep canary size."""
import _setup
import numpy as np, pandas as pd
from pipeline import run_one_seed
from metrics import catch_and_precision

rows = []
for size in [100, 150, 300, 600, 1200]:
    for seed in range(1, 11):
        ps, scores, _ = run_one_seed(seed, canary_size=size, methods=["structural_lr", "combined_rf"])
        sf = (ps.ground_truth_pass == 0).to_numpy(); rng = np.random.default_rng(seed)
        for m, sc in scores.items():
            rows.append({"canary_size": size, "seed": seed, "method": m,
                         "catch": catch_and_precision(sc, sf, .2, rng)[0]})
s = pd.DataFrame(rows).groupby(["canary_size", "method"])["catch"].agg(["mean", "std"]).reset_index()
s.to_csv(_setup.RESULTS / "canary_size.csv", index=False)
print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
