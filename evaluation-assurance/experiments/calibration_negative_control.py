"""
1. Calibration of the probabilities (matters if they feed a cost formula).
2. Negative control: adding a pure-noise feature must not improve catch.
3. Independence: correlation between evaluator confidence and auditor score.
"""
import _setup
import numpy as np, pandas as pd
from pipeline import split
from audit_models import Auditor
from metrics import ece, reliability_table, catch_and_precision

cal, nc, corr = [], [], []
for seed in range(1, _setup.N_SEEDS + 1):
    canary, ps = split(seed)
    y = ps.ground_truth_pass.to_numpy(); sf = y == 0
    for kind in ["lr", "rf"]:
        cal.append({"seed": seed, "model": kind,
                    "ece": ece(Auditor("structural", kind, seed).fit(canary).score(ps), y)})
    base = Auditor("structural", "lr", seed).fit(canary).score(ps)
    noisy = Auditor("structural", "lr", seed, include_noise_control=True).fit(canary).score(ps)
    rng = np.random.default_rng(seed)
    nc.append({"seed": seed, "without": catch_and_precision(base, sf, .2, rng)[0],
               "with_noise": catch_and_precision(noisy, sf, .2, rng)[0]})
    corr.append(np.corrcoef(ps.evaluator_confidence, base)[0, 1])

cal = pd.DataFrame(cal).groupby("model").ece.agg(["mean", "std"])
nc = pd.DataFrame(nc); nc["diff"] = nc.with_noise - nc.without
print("ECE (lower = better calibrated):\n", cal.round(3))
print(f"\nNegative control: catch without noise {nc.without.mean():.1%}, with noise "
      f"{nc.with_noise.mean():.1%}, diff {nc['diff'].mean():+.1%} ± {1.96*nc['diff'].std()/np.sqrt(len(nc)):.1%}")
print(f"Correlation evaluator-confidence vs auditor score: {np.mean(corr):.2f}")
canary, ps = split(42)
a = Auditor("structural", "lr", 42).fit(canary)
print("\nReliability table (seed 42, logistic):")
print(reliability_table(a.score(ps), ps.ground_truth_pass).round(3).to_string(index=False))
pd.DataFrame({"model": cal.index, "ece_mean": cal["mean"].values}).to_csv(_setup.RESULTS / "calibration.csv", index=False)
nc.to_csv(_setup.RESULTS / "negative_control.csv", index=False)
