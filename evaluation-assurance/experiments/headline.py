"""
Headline comparison across 20 seeds: for the same human-review budget,
how many silent failures does each method catch, and how many of the
escalations are real failures?
"""
import _setup
import numpy as np, pandas as pd
from pipeline import run_one_seed, ALL_METHODS, LABELS
from metrics import catch_and_precision

rows = []
for seed in range(1, _setup.N_SEEDS + 1):
    pass_set, scores, _ = run_one_seed(seed)
    sf = (pass_set.ground_truth_pass == 0).to_numpy()
    rng = np.random.default_rng(10_000 + seed)
    for m in ALL_METHODS:
        for b in _setup.BUDGETS:
            c, p = catch_and_precision(scores[m], sf, b, rng)
            rows.append({"seed": seed, "method": m, "budget": b, "catch": c, "precision": p,
                         "silent_failure_rate": sf.mean(), "n_pass": len(sf)})
df = pd.DataFrame(rows)
df.to_csv(_setup.RESULTS / "headline_by_seed.csv", index=False)

s = df.groupby(["method", "budget"]).agg(catch=("catch", "mean"), catch_sd=("catch", "std"),
                                         precision=("precision", "mean")).reset_index()
s["catch_ci95"] = 1.96 * s.catch_sd / np.sqrt(_setup.N_SEEDS)
s["lift_vs_random"] = s["catch"] / s["budget"]
s.to_csv(_setup.RESULTS / "headline_summary.csv", index=False)

print(f"Silent-failure rate among evaluator PASS verdicts: {df.silent_failure_rate.mean():.1%}\n")
for b in _setup.BUDGETS:
    print(f"--- review budget {b:.0%} of PASS verdicts ---")
    for m in ALL_METHODS:
        r = s[(s.method == m) & (s.budget == b)].iloc[0]
        print(f"  {LABELS[m]:48s} catch {r['catch']:5.1%} ± {r.catch_ci95:4.1%}   "
              f"precision {r.precision:5.1%}   lift {r.lift_vs_random:.2f}x")
