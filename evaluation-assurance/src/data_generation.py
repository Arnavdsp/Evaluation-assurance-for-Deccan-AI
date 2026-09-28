"""
Synthetic agent traces with a known ground truth.

The rule that fixed my v2 leakage bug: every failure mode has a latent cause
(latent_* columns) that no model ever sees. Models only get a noisy, partial,
sometimes-missing reading of it (obs_* columns), which is roughly what real
logging gives you. verify_no_leakage() enforces this.

Failure modes:
  A  edge-case regression   rare code path breaks, low test coverage
  B  state inconsistency    agent says it saved, the DB disagrees
  C  wrong tool argument    tool call "works" but on the wrong object
  D  partial completion     some required actions silently skipped
  E  ordering violation     end state looks fine, steps ran out of order

ground_truth_pass = final_suite_pass and no mode triggered.

noise_feature is pure noise, only there for the negative control.
"""

import numpy as np
import pandas as pd

DEFAULT_N = 6000


def generate(n=DEFAULT_N, seed=42, complexity_lo=0.0, complexity_hi=1.0,
             rare_path_base=0.15, rare_path_slope=0.25):
    # complexity / rare-path knobs are only changed for the distribution-shift test
    rng = np.random.default_rng(seed)

    # task metadata, visible to everyone
    task_complexity = rng.uniform(complexity_lo, complexity_hi, n)
    num_steps = rng.poisson(5 + 15 * task_complexity, n).clip(1, None)
    tool_call_count = rng.poisson(3 + 10 * task_complexity, n).clip(0, None)
    latency_variance = rng.gamma(2.0, 1.0 + 2.0 * task_complexity, n)

    surface_pass_prob = np.clip(0.93 - 0.10 * task_complexity, 0.05, 0.99)
    final_suite_pass = rng.binomial(1, surface_pass_prob)

    # A: rare path exercised + low true coverage
    latent_rare_path = rng.binomial(1, np.clip(rare_path_base + rare_path_slope * task_complexity, 0, 1))
    cov_alpha = np.clip(6 - 3.5 * task_complexity, 0.5, None)
    cov_beta = np.clip(2 + 1.5 * task_complexity, 0.5, None)
    latent_true_coverage = rng.beta(cov_alpha, cov_beta)
    latent_mode_a = rng.binomial(1, np.clip(0.65 * latent_rare_path * (1 - latent_true_coverage), 0, 0.95))
    # what we observe: noisy coverage reading (missing 12% of the time)
    # and a rare-path flag that's wrong 6% of the time
    coverage_noise = rng.normal(0, 0.12, n)
    obs_coverage_reading = np.clip(latent_true_coverage + coverage_noise, 0, 1)
    coverage_missing = rng.binomial(1, 0.12, n).astype(bool)
    obs_coverage_reading = np.where(coverage_missing, np.nan, obs_coverage_reading)
    rare_path_flip = rng.binomial(1, 0.06, n).astype(bool)
    obs_rare_path_flag = np.where(rare_path_flip, 1 - latent_rare_path, latent_rare_path)

    # B: agent says the write worked, DB disagrees. more likely with many tool calls
    state_mismatch_prob = np.clip(0.015 + 0.10 * (tool_call_count > 8), 0, 1)
    latent_mode_b = rng.binomial(1, state_mismatch_prob)
    # reconciliation check: runs on 85% of traces, misses 10% of real mismatches
    recon_ran = rng.binomial(1, 0.85, n).astype(bool)
    recon_false_neg = rng.binomial(1, 0.10, n).astype(bool)
    recon_flags_inconsistency = latent_mode_b.astype(bool) & ~recon_false_neg
    obs_state_recon_flag = np.where(recon_ran, recon_flags_inconsistency.astype(float), np.nan)

    # C: tool call succeeded on the wrong object
    wrong_arg_prob = np.clip(0.01 + 0.006 * tool_call_count, 0, 0.5)
    latent_mode_c = rng.binomial(1, wrong_arg_prob)
    # argument audit: only logged for 55% of traces, and it's a noisy score, not a flag
    audit_logged = rng.binomial(1, 0.55, n).astype(bool)
    audit_score = np.clip(rng.normal(0.75 - 0.5 * latent_mode_c, 0.18, n), 0, 1)
    obs_tool_arg_audit_score = np.where(audit_logged, audit_score, np.nan)

    # D: 1-2 of 5 required actions silently skipped (kept rare on purpose,
    # otherwise it's not a *silent* failure)
    required_actions = 5
    shortfall_trigger_prob = np.clip(0.02 + 0.05 * task_complexity, 0, 1)
    latent_mode_d = rng.binomial(1, shortfall_trigger_prob)
    deficit = rng.integers(1, 3, n)
    actions_done = np.clip(required_actions - latent_mode_d * deficit, 0, required_actions)
    # checklist: coarse bucket from a noisy re-count, unknown 5% of the time
    noisy_actions_done = np.clip(actions_done + rng.normal(0, 0.6, n), 0, required_actions)
    frac = noisy_actions_done / required_actions
    completion_bucket = np.select([frac >= 0.99, frac >= 0.75], ["all_done", "most_done"],
                                  default="partial")
    bucket_missing = rng.binomial(1, 0.05, n).astype(bool)
    completion_bucket = np.where(bucket_missing, "unknown", completion_bucket)

    # E: ordering rule broken mid-trace, end state looks fine. more steps -> more risk
    ordering_violation_prob = np.clip(0.01 + 0.003 * num_steps, 0, 0.4)
    latent_mode_e = rng.binomial(1, ordering_violation_prob)
    # replay check is expensive: runs on 35% of traces, misses 8%
    replay_ran = rng.binomial(1, 0.35, n).astype(bool)
    replay_miss = rng.binomial(1, 0.08, n).astype(bool)
    replay_flags_violation = latent_mode_e.astype(bool) & ~replay_miss
    obs_replay_ordering_flag = np.where(replay_ran, replay_flags_violation.astype(float), np.nan)

    noise_feature = rng.normal(0, 1, n)  # negative control

    any_mode = (latent_mode_a | latent_mode_b | latent_mode_c | latent_mode_d | latent_mode_e).astype(bool)
    ground_truth_pass = ((final_suite_pass == 1) & ~any_mode).astype(int)

    df = pd.DataFrame({
        "task_id": [f"T{100000+i}" for i in range(n)],
        # metadata
        "task_complexity": task_complexity.round(3),
        "num_steps": num_steps,
        "tool_call_count": tool_call_count,
        "latency_variance": latency_variance.round(3),
        "final_suite_pass": final_suite_pass,
        # observed signals (auditor only)
        "obs_rare_path_flag": obs_rare_path_flag,
        "obs_coverage_reading": np.round(obs_coverage_reading, 3),
        "obs_state_recon_flag": obs_state_recon_flag,
        "obs_tool_arg_audit_score": np.round(obs_tool_arg_audit_score, 3),
        "obs_completion_bucket": completion_bucket,
        "obs_replay_ordering_flag": obs_replay_ordering_flag,
        "noise_feature": noise_feature.round(3),
        # latent truth: only used for scoring, never as a feature
        "latent_mode_a": latent_mode_a,
        "latent_mode_b": latent_mode_b,
        "latent_mode_c": latent_mode_c,
        "latent_mode_d": latent_mode_d,
        "latent_mode_e": latent_mode_e,
        "ground_truth_pass": ground_truth_pass,
    })
    return df


LATENT_COLUMNS = ["latent_mode_a", "latent_mode_b", "latent_mode_c",
                   "latent_mode_d", "latent_mode_e", "ground_truth_pass"]


def verify_no_leakage(feature_list):
    # cheap guard so the v2 bug can't sneak back in
    leaked = [c for c in feature_list if c in LATENT_COLUMNS]
    if leaked:
        raise ValueError(f"LEAKAGE: latent columns in feature list: {leaked}")


if __name__ == "__main__":
    df = generate()
    print(df.head(8).to_string(index=False))
    print(f"\nN = {len(df)}")
    print(f"Ground-truth pass rate: {df.ground_truth_pass.mean():.1%}")
    for m in ["a", "b", "c", "d", "e"]:
        col = f"latent_mode_{m}"
        print(f"  mode {m} trigger rate: {df[col].mean():.1%}")
    print(f"Missingness -- coverage_reading: {df.obs_coverage_reading.isna().mean():.1%}, "
          f"state_recon: {df.obs_state_recon_flag.isna().mean():.1%}, "
          f"tool_arg_audit: {df.obs_tool_arg_audit_score.isna().mean():.1%}, "
          f"replay_ordering: {df.obs_replay_ordering_flag.isna().mean():.1%}")
