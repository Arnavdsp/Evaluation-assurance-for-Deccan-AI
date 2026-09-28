"""
(1) Robustness: make the logged signals noisier / sparser than modeled.
(2) Distribution shift: train on one task mix, evaluate on a harder one.
Synthetic-to-synthetic only -- real shift behaviour is untested.
"""
import _setup
import numpy as np, pandas as pd
from pipeline import split
from audit_models import Auditor
from metrics import catch_and_precision


def perturb(df, noise, miss, flip, seed):
    rng = np.random.default_rng(seed + 9000); out = df.copy()
    for c in ["obs_coverage_reading", "obs_tool_arg_audit_score"]:
        out[c] = np.clip(out[c] + rng.normal(0, noise, len(out)), 0, 1)
        out.loc[rng.random(len(out)) < miss, c] = np.nan
    for c in ["obs_state_recon_flag", "obs_replay_ordering_flag"]:
        f = (rng.random(len(out)) < flip) & out[c].notna()
        out.loc[f, c] = 1 - out.loc[f, c]
        out.loc[rng.random(len(out)) < miss, c] = np.nan
    f = rng.random(len(out)) < flip
    out.loc[f, "obs_rare_path_flag"] = 1 - out.loc[f, "obs_rare_path_flag"]
    return out

LEVELS = [("as modeled", 0, 0, 0), ("mild", .08, .05, .03), ("moderate", .16, .12, .08), ("severe", .30, .25, .18)]
rows = []
for seed in range(1, _setup.N_SEEDS + 1):
    canary, ps = split(seed)
    a = Auditor("structural", "lr", seed).fit(canary)
    rng = np.random.default_rng(seed)
    for name, n, m, f in LEVELS:
        p = perturb(ps, n, m, f, seed)
        rows.append({"seed": seed, "test": "robustness", "level": name,
                     "catch": catch_and_precision(a.score(p), p.ground_truth_pass == 0, .2, rng)[0]})
    _, shifted = split(seed + 500, complexity_lo=0.35, rare_path_base=0.30, rare_path_slope=0.30)
    rows.append({"seed": seed, "test": "shift", "level": "harder task mix",
                 "catch": catch_and_precision(a.score(shifted), shifted.ground_truth_pass == 0, .2, rng)[0]})
s = pd.DataFrame(rows).groupby(["test", "level"], sort=False)["catch"].agg(["mean", "std"]).reset_index()
s.to_csv(_setup.RESULTS / "robustness_shift.csv", index=False)
print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
