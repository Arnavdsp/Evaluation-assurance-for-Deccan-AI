"""
All the ways I compare for picking which PASS verdicts a human should re-check.
Every scorer returns something like P(truly passed), so lower = more suspicious.

  random        - spot-check baseline
  rules         - count how many checks fired, no training
  surface       - model on the same info the evaluator had
  structural_lr - logistic regression on the execution checks (what I'd use)
  structural_rf / combined_rf - random forest versions, for comparison

Learned models are trained on a small labelled sample of PASS verdicts
("canary set").
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from data_generation import verify_no_leakage
from evaluator_mock import SURFACE_FEATURES

BUCKETS = ["all_done", "most_done", "partial", "unknown"]


def structural_features(df: pd.DataFrame, include_noise_control=False) -> pd.DataFrame:
    # missing checks get a neutral fill + a "was missing" flag, since
    # "the check didn't run" is itself informative
    out = pd.DataFrame(index=df.index)
    out["obs_rare_path_flag"] = df["obs_rare_path_flag"].astype(float)
    for col in ["obs_coverage_reading", "obs_tool_arg_audit_score"]:
        out[f"{col}_missing"] = df[col].isna().astype(float)
        out[col] = df[col].fillna(df[col].median())
    for col in ["obs_state_recon_flag", "obs_replay_ordering_flag"]:
        out[f"{col}_missing"] = df[col].isna().astype(float)
        out[col] = df[col].fillna(0.5)
    for b in BUCKETS:
        out[f"bucket_{b}"] = (df["obs_completion_bucket"] == b).astype(float)
    if include_noise_control:
        out["noise_feature"] = df["noise_feature"].astype(float)
    return out


def build_features(df: pd.DataFrame, feature_set: str, include_noise_control=False) -> pd.DataFrame:
    surface = df[SURFACE_FEATURES].astype(float)
    if feature_set == "surface":
        feat = surface
    elif feature_set == "structural":
        feat = structural_features(df, include_noise_control)
    elif feature_set == "combined":
        feat = pd.concat([surface, structural_features(df, include_noise_control)], axis=1)
    else:
        raise ValueError(feature_set)
    verify_no_leakage(list(feat.columns))
    return feat


def make_model(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                      random_state=seed)
    if kind == "lr":
        return LogisticRegression(max_iter=2000, C=1.0)
    raise ValueError(kind)


class Auditor:
    # feature set + model type

    def __init__(self, feature_set: str, kind: str, seed: int = 0, include_noise_control=False):
        self.feature_set, self.kind, self.seed = feature_set, kind, seed
        self.include_noise_control = include_noise_control

    def fit(self, canary: pd.DataFrame):
        X = build_features(canary, self.feature_set, self.include_noise_control)
        self.columns_ = list(X.columns)
        self.model_ = make_model(self.kind, self.seed).fit(X, canary["ground_truth_pass"])
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = build_features(df, self.feature_set, self.include_noise_control)
        X = X.reindex(columns=self.columns_, fill_value=0.0)
        return self.model_.predict_proba(X)[:, 1]

    def reason_codes(self) -> pd.Series:
        # LR weights per signal; negative = pushes towards suspicious.
        # This is what a reviewer would see as "why was this flagged".
        if self.kind != "lr":
            raise ValueError("reason codes are only defined for the logistic model")
        return pd.Series(self.model_.coef_[0], index=self.columns_).sort_values()


def rules_score(df: pd.DataFrame) -> np.ndarray:
    # no-ML baseline: how many checks fired (+ small tie-breaker).
    # negated so lower = more suspicious, same as the models
    red = ((df["obs_state_recon_flag"] == 1).astype(float)
           + (df["obs_replay_ordering_flag"] == 1).astype(float)
           + df["obs_completion_bucket"].isin(["partial", "most_done"]).astype(float)
           + (df["obs_tool_arg_audit_score"] < 0.5).astype(float)
           + ((df["obs_rare_path_flag"] == 1) & (df["obs_coverage_reading"] < 0.5)).astype(float))
    tie = (0.1 * (1 - df["obs_tool_arg_audit_score"].fillna(0.75))
           + 0.1 * df["obs_rare_path_flag"] * (1 - df["obs_coverage_reading"].fillna(0.7)))
    return -(red + tie).to_numpy()


def random_score(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(size=n)
