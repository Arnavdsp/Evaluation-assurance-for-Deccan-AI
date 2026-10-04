import pytest

from audit_models import build_features
from data_generation import LATENT_COLUMNS, generate, verify_no_leakage
from metrics import catch_and_precision
from pipeline import run_one_seed


def test_verify_no_leakage_rejects_latent_columns():
    with pytest.raises(ValueError, match="LEAKAGE"):
        verify_no_leakage(["obs_coverage_reading", LATENT_COLUMNS[0]])


@pytest.mark.parametrize("feature_set", ["surface", "structural", "combined"])
def test_feature_sets_contain_no_latent_columns(feature_set):
    features = build_features(generate(n=300, seed=0), feature_set)
    assert not set(features.columns) & set(LATENT_COLUMNS)
    assert not features.isna().any().any()


def test_generation_is_deterministic_per_seed():
    a, b = generate(n=200, seed=3), generate(n=200, seed=3)
    assert a.equals(b)


def test_execution_checks_beat_random_on_one_seed():
    # small, single-seed version of the headline result; the full 20-seed
    # numbers come from experiments/headline.py
    pass_set, scores, _ = run_one_seed(seed=0, n=3000, methods=["random", "rules", "structural_lr"])
    failed = pass_set["ground_truth_pass"] == 0
    catch = {m: catch_and_precision(s, failed, 0.2)[0] for m, s in scores.items()}
    assert catch["structural_lr"] > catch["random"] + 0.1
    assert catch["rules"] > catch["random"]
