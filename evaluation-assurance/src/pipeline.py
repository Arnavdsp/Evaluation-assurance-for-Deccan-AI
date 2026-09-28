"""
One full run for one seed: generate traces -> evaluator verdicts -> label a
small sample of PASS verdicts -> fit auditors -> score the other PASS verdicts.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from data_generation import generate
from evaluator_mock import evaluator_mock_verdict
from audit_models import Auditor, rules_score, random_score

LEARNED = {
    "surface_rf":    ("surface", "rf"),
    "structural_lr": ("structural", "lr"),
    "structural_rf": ("structural", "rf"),
    "combined_rf":   ("combined", "rf"),
}
ALL_METHODS = ["random", "rules"] + list(LEARNED)

LABELS = {
    "random":        "Random spot-check",
    "rules":         "Rule: count fired checks (no ML)",
    "surface_rf":    "Model on the evaluator's own info",
    "structural_lr": "Independent signals + logistic reg. (proposed)",
    "structural_rf": "Independent signals + random forest",
    "combined_rf":   "Both + random forest",
}


def split(seed, n=6000, canary_size=300, **gen_kwargs):
    df = evaluator_mock_verdict(generate(n=n, seed=seed, **gen_kwargs), seed=seed)
    canary, rest = train_test_split(df, train_size=canary_size, random_state=seed,
                                    stratify=df["ground_truth_pass"])
    return canary[canary.evaluator_verdict == 1], rest[rest.evaluator_verdict == 1]


def run_one_seed(seed, n=6000, canary_size=300, methods=ALL_METHODS, **gen_kwargs):
    canary_pass, pass_set = split(seed, n, canary_size, **gen_kwargs)
    rng = np.random.default_rng(seed)
    scores, auditors = {}, {}
    for m in methods:
        if m == "random":
            scores[m] = random_score(len(pass_set), rng)
        elif m == "rules":
            scores[m] = rules_score(pass_set)
        else:
            fs, kind = LEARNED[m]
            a = Auditor(fs, kind, seed).fit(canary_pass)
            scores[m], auditors[m] = a.score(pass_set), a
    return pass_set, scores, auditors
