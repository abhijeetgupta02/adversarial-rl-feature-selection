"""Tests for the standalone round-2 diagnostic helpers.

These helpers (`sturges_bins`, `freedman_diaconis_bins`,
`majority_class_baseline`) are intentionally NOT wired into `evaluate`, so they
cannot move the reconstructed metrics in `reports/latest/`. Each is cross-checked
against an independent reference (NumPy's own bin estimators / scikit-learn's
DummyClassifier) plus hand-computed values, and the `feature_descriptors`
empty-window guard is exercised on its error path.
"""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from robustfeatures.core import (
    feature_descriptors,
    freedman_diaconis_bins,
    inject_imposters,
    majority_class_baseline,
    sturges_bins,
)


# --- Sturges' rule --------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 10, 50, 100, 1000, 3601])
def test_sturges_bins_matches_numpy(n):
    # NumPy's "sturges" estimator is an independent implementation of the rule.
    sample = np.random.default_rng(n).normal(size=n)
    numpy_bins = len(np.histogram_bin_edges(sample, bins="sturges")) - 1
    assert sturges_bins(n) == numpy_bins


def test_sturges_bins_small_and_monotone():
    assert sturges_bins(1) == 1  # ceil(log2(1)) + 1
    counts = [sturges_bins(n) for n in range(1, 200)]
    assert counts == sorted(counts)  # non-decreasing in n
    assert all(isinstance(c, int) for c in counts)


def test_sturges_bins_rejects_nonpositive():
    with pytest.raises(ValueError):
        sturges_bins(0)
    with pytest.raises(ValueError):
        sturges_bins(-5)


# --- Freedman-Diaconis rule ----------------------------------------------


@pytest.mark.parametrize("n", [20, 50, 200, 1000, 4000])
def test_freedman_diaconis_matches_numpy(n):
    sample = np.random.default_rng(n).normal(size=n)
    numpy_bins = len(np.histogram_bin_edges(sample, bins="fd")) - 1
    assert freedman_diaconis_bins(sample) == numpy_bins


def test_freedman_diaconis_constant_input_falls_back_to_one_bin():
    # Zero IQR / zero range must not divide by zero; one bin is the safe floor.
    assert freedman_diaconis_bins(np.ones(100)) == 1
    assert freedman_diaconis_bins(np.zeros(3)) == 1


def test_freedman_diaconis_returns_positive_int_and_accepts_2d():
    sample = np.random.default_rng(0).normal(size=(500, 3))
    bins = freedman_diaconis_bins(sample)  # ravels internally
    assert isinstance(bins, int) and bins >= 1


def test_freedman_diaconis_rejects_empty():
    with pytest.raises(ValueError):
        freedman_diaconis_bins(np.array([]))


# --- majority-class baseline ---------------------------------------------


def test_majority_class_baseline_matches_dummy_classifier():
    rng = np.random.default_rng(7)
    for _ in range(5):
        labels = rng.integers(0, 3, size=200)
        features = np.zeros((labels.size, 1))
        dummy = DummyClassifier(strategy="most_frequent").fit(features, labels)
        reference = float((dummy.predict(features) == labels).mean())
        assert majority_class_baseline(labels) == pytest.approx(reference)


def test_majority_class_baseline_hand_values():
    assert majority_class_baseline(np.array([1, 1, 1, -1])) == pytest.approx(0.75)
    assert majority_class_baseline(np.array([1, -1, 1, -1])) == pytest.approx(0.5)
    assert majority_class_baseline(np.array([5, 5, 5])) == pytest.approx(1.0)


def test_majority_class_baseline_is_a_floor_on_descriptor_labels():
    # On the descriptor label vector the floor is in (0, 1] and never exceeds 1.
    clean = np.random.default_rng(7).normal(size=(600, 8))
    _injected, labels = inject_imposters(clean, 2, "gaussian", 7)
    floor = majority_class_baseline(labels)
    assert 0.0 < floor <= 1.0
    # 8 originals (-1) vs 2 imposters (+1) -> majority is the originals.
    assert floor == pytest.approx(8 / 10)


def test_majority_class_baseline_rejects_empty():
    with pytest.raises(ValueError):
        majority_class_baseline(np.array([]))


# --- feature_descriptors window guard ------------------------------------


def test_feature_descriptors_rejects_windows_exceeding_rows():
    clean = np.random.default_rng(0).normal(size=(5, 8))
    injected, labels = inject_imposters(clean, 2, "gaussian", 7)
    with pytest.raises(ValueError, match="windows"):
        feature_descriptors(injected, labels, windows=12)  # 12 > 5 rows


def test_feature_descriptors_rejects_nonpositive_windows():
    clean = np.random.default_rng(0).normal(size=(20, 8))
    injected, labels = inject_imposters(clean, 2, "gaussian", 7)
    with pytest.raises(ValueError):
        feature_descriptors(injected, labels, windows=0)


def test_feature_descriptors_allows_windows_equal_to_rows():
    # Boundary: each window holds exactly one row; descriptors stay well-defined.
    clean = np.random.default_rng(0).normal(size=(6, 8))
    injected, labels = inject_imposters(clean, 2, "gaussian", 7)
    descriptors = feature_descriptors(injected, labels, windows=6)
    assert len(descriptors) == 6 * injected.shape[1]
