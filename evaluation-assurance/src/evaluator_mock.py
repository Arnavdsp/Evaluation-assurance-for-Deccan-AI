"""
Stand-in for the primary evaluator (think LLM judge).

It only sees task metadata + final_suite_pass, i.e. roughly "did the output
look right". It never sees the obs_* execution checks.
"""
import numpy as np
import pandas as pd

SURFACE_FEATURES = ["task_complexity", "num_steps", "tool_call_count",
                    "latency_variance", "final_suite_pass"]


def evaluator_mock_verdict(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)  # separate stream from data generation
    noise = rng.normal(0, 0.05, len(df))
    confidence = (0.90 * df["final_suite_pass"] + 0.05 * (1 - df["task_complexity"]) + noise).clip(0, 1)
    out = df.copy()
    out["evaluator_confidence"] = confidence.round(3)
    out["evaluator_verdict"] = (confidence >= 0.5).astype(int)  # 1 = PASS
    return out


helix_mock_verdict = evaluator_mock_verdict  # old name, kept so old scripts still run
