"""
Metrics, framed as: "a human can re-check X% of PASS verdicts. which X%?"

catch     = share of silent failures that end up in the reviewed X%
precision = share of reviewed cases that really were failures
lift      = catch / X  (random review = 1.0)
"""

import numpy as np


def review_top(scores, budget, rng=None):
    # flag exactly budget*n lowest scores. tiny jitter breaks ties so the
    # rule (lots of tied scores) doesn't get lucky from sort order
    s = np.asarray(scores, dtype=float)
    if rng is not None:
        s = s + rng.uniform(0, 1e-9, len(s))
    k = int(round(budget * len(s)))
    flagged = np.zeros(len(s), dtype=bool)
    flagged[np.argsort(s, kind="stable")[:k]] = True
    return flagged


def catch_and_precision(scores, is_silent_failure, budget, rng=None):
    sf = np.asarray(is_silent_failure, dtype=bool)
    flagged = review_top(scores, budget, rng)
    caught = int((flagged & sf).sum())
    return caught / sf.sum(), caught / max(flagged.sum(), 1)


def curve(scores, is_silent_failure, budgets=np.linspace(0, 1, 51), rng=None):
    return np.array([catch_and_precision(scores, is_silent_failure, b, rng)[0] if b > 0 else 0.0
                     for b in budgets])


def ece(p, y, n_bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


def reliability_table(p, y, n_bins=10):
    import pandas as pd
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bucket": f"{b/n_bins:.1f}-{(b+1)/n_bins:.1f}", "n": int(m.sum()),
                         "mean_predicted": p[m].mean(), "observed_pass_rate": y[m].mean()})
    return pd.DataFrame(rows)
