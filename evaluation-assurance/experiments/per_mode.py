"""
Catch rate per failure mode vs. the instrumentation ceiling (share of
that mode's failures where its check actually ran). Separates "model is
weak" from "the signal was never logged".
"""
import _setup
import numpy as np, pandas as pd
from pipeline import run_one_seed
from metrics import review_top

MODES = {"a": ("Edge-case regression", ["obs_coverage_reading"]),
         "b": ("State inconsistency", ["obs_state_recon_flag"]),
         "c": ("Wrong tool argument", ["obs_tool_arg_audit_score"]),
         "d": ("Partial completion", None),
         "e": ("Ordering violation", ["obs_replay_ordering_flag"])}
rows = []
for seed in range(1, _setup.N_SEEDS + 1):
    ps, scores, _ = run_one_seed(seed, methods=["structural_lr"])
    flagged = review_top(scores["structural_lr"], 0.20, np.random.default_rng(seed))
    sf = (ps.ground_truth_pass == 0).to_numpy()
    for k, (name, cols) in MODES.items():
        m = sf & (ps[f"latent_mode_{k}"] == 1).to_numpy()
        if k == "d":
            observed = (ps["obs_completion_bucket"] != "unknown").to_numpy()
        else:
            observed = ps[cols[0]].notna().to_numpy()
        rows.append({"seed": seed, "mode": k.upper(), "name": name, "n": m.sum(),
                     "catch": flagged[m].mean(), "ceiling": observed[m].mean()})
s = pd.DataFrame(rows).groupby(["mode", "name"])[["n", "catch", "ceiling"]].mean().reset_index()
s.to_csv(_setup.RESULTS / "per_mode.csv", index=False)
print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
