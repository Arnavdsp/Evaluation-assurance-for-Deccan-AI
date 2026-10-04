import numpy as np
import pytest

from metrics import catch_and_precision, curve, ece, review_top


def test_review_top_flags_the_lowest_scores():
    flagged = review_top([0.9, 0.1, 0.5, 0.3, 0.7], budget=0.4)
    assert flagged.tolist() == [False, True, False, True, False]


def test_review_top_rounds_the_budget_to_a_whole_count():
    assert review_top(np.arange(10), budget=0.25).sum() == 2  # round(2.5) == 2
    assert review_top(np.arange(10), budget=0.0).sum() == 0
    assert review_top(np.arange(10), budget=1.0).sum() == 10


def test_catch_and_precision():
    scores = [0.1, 0.2, 0.3, 0.8, 0.9]
    failed = [True, False, True, True, False]
    catch, precision = catch_and_precision(scores, failed, budget=0.4)  # flags the first two
    assert catch == pytest.approx(1 / 3)
    assert precision == pytest.approx(1 / 2)


def test_random_scores_catch_about_the_budget():
    rng = np.random.default_rng(0)
    failed = rng.random(20_000) < 0.25
    catch, precision = catch_and_precision(rng.random(20_000), failed, budget=0.2)
    assert catch == pytest.approx(0.2, abs=0.02)
    assert precision == pytest.approx(0.25, abs=0.02)


def test_curve_is_non_decreasing_and_ends_at_one():
    rng = np.random.default_rng(1)
    failed = rng.random(500) < 0.3
    c = curve(rng.random(500), failed)
    assert c[0] == 0.0
    assert np.all(np.diff(c) >= 0)
    assert c[-1] == pytest.approx(1.0)


def test_ece_is_zero_when_calibrated_and_positive_when_not():
    p = np.repeat([0.25, 0.75], 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert ece(p, y) == pytest.approx(0.0)
    assert ece(np.full(8, 0.9), np.zeros(8)) == pytest.approx(0.9)
