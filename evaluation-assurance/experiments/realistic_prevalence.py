"""
The synthetic world has ~25% of PASS verdicts silently wrong -- far too
high for a production evaluator. Here we subsample silent failures down
to 5% and 2% of PASS verdicts and re-measure (models unchanged).
Catch rate stays useful; precision (reviewer yield) drops -- that is the
number that decides reviewer cost.
"""
import _setup
import numpy as np, pandas as pd
from pipeline import run_one_seed
from metrics import catch_and_precision

rows = []
for seed in range(1, _setup.N_SEEDS + 1):
    pass_set, scores, _ = run_one_seed(seed, methods=["random", "rules", "structural_lr"])
    sf = (pass_set.ground_truth_pass == 0).to_numpy()
    rng = np.random.default_rng(20_000 + seed)
    ok, bad = np.where(~sf)[0], np.where(sf)[0]
    for prev in [0.05, 0.02]:
        pick = np.concatenate([ok, rng.choice(bad, int(prev / (1 - prev) * len(ok)), replace=False)])
        for m, sc in scores.items():
            for b in [0.05, 0.10, 0.20]:
                c, p = catch_and_precision(sc[pick], sf[pick], b, rng)
                rows.append({"seed": seed, "prevalence": prev, "method": m, "budget": b,
                             "catch": c, "precision": p})
df = pd.DataFrame(rows)
s = df.groupby(["prevalence", "method", "budget"])[["catch", "precision"]].mean().reset_index()
s.to_csv(_setup.RESULTS / "realistic_prevalence.csv", index=False)
print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
