import numpy as np
import pytest

from robustfeatures.core import (
    ENVIRONMENTS,
    benchmark_diagnostics,
    deviation_gate,
    entropy,
    evaluate,
    feature_descriptors,
    inject_imposters,
    joint_entropy,
    kl_divergence,
    models,
    mutual_information,
    synthetic_observations,
)


@pytest.fixture(scope="module")
def evaluation():
    """Single shared evaluation run reused across the heavier tests."""
    return evaluate(samples=360, seed=7)


# --- information measures -------------------------------------------------


def test_entropy_constant_is_zero_and_spread_is_positive():
    assert entropy(np.zeros(500)) == 0.0
    assert entropy(np.random.default_rng(7).normal(size=2000)) > 0


def test_information_measures_are_finite():
    rng = np.random.default_rng(7)
    left, right = rng.normal(size=1000), rng.uniform(size=1000)
    assert entropy(left) > 0
    assert joint_entropy(left, right) > 0
    assert np.isfinite(kl_divergence(left, right))


def test_kl_divergence_self_is_zero_and_nonnegative():
    rng = np.random.default_rng(7)
    sample = rng.normal(size=2000)
    assert kl_divergence(sample, sample) == pytest.approx(0.0, abs=1e-9)
    other = rng.normal(3.0, 1.0, size=2000)
    assert kl_divergence(sample, other) > 0


def test_mutual_information_nonnegative_and_tracks_dependence():
    rng = np.random.default_rng(7)
    sample = rng.normal(size=4000)
    independent = rng.normal(size=4000)
    # Perfectly dependent (X vs X) shares more information than independent draws.
    assert mutual_information(sample, sample) > mutual_information(sample, independent)
    # Plug-in MI from consistent marginals is always non-negative.
    assert mutual_information(sample, independent) >= 0.0
    assert mutual_information(sample, sample) > 0.0


def test_mutual_information_empty_input_is_zero():
    empty = np.array([])
    assert mutual_information(empty, empty) == 0.0


# --- imposter injection ---------------------------------------------------


def test_imposters_are_appended_and_labeled():
    clean = np.ones((100, 8))
    injected, labels = inject_imposters(clean, 4, "uniform", 7)
    assert injected.shape == (100, 12)
    assert labels.tolist() == [-1] * 8 + [1] * 4


def test_gaussian_injection_shape_and_labels():
    clean = np.ones((100, 8))
    injected, labels = inject_imposters(clean, 3, "gaussian", 7)
    assert injected.shape == (100, 11)
    assert labels.tolist() == [-1] * 8 + [1] * 3


def test_out_of_range_uniform_has_wider_support_than_in_range():
    clean = np.zeros((4000, 4))
    in_range, _ = inject_imposters(clean, 4, "uniform", 7, in_range=True)
    out_range, _ = inject_imposters(clean, 4, "uniform", 7, in_range=False)
    assert np.abs(in_range[:, -4:]).max() <= 1.0 + 1e-9
    assert np.abs(out_range[:, -4:]).max() > 1.5


def test_unsupported_noise_raises():
    with pytest.raises(ValueError):
        inject_imposters(np.ones((10, 3)), 1, "poisson", 7)


# --- fixtures / descriptors ----------------------------------------------


def test_synthetic_observations_shape_and_determinism():
    spec = ENVIRONMENTS["lunar_lander"]
    first = synthetic_observations(spec, 200, seed=7)
    second = synthetic_observations(spec, 200, seed=7)
    different = synthetic_observations(spec, 200, seed=8)
    assert first.shape == (200, spec.dimensions)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_algorithm_feature_matrix_and_gate():
    clean = np.random.default_rng(7).normal(size=(600, 8))
    injected, labels = inject_imposters(clean, 2, "gaussian", 7)
    descriptors = feature_descriptors(injected, labels)
    assert {"mean_entropy", "centered_entropy", "entropy", "joint_entropy", "kl"}.issubset(
        descriptors
    )
    assert deviation_gate(301, ENVIRONMENTS["lunar_lander"])
    assert not deviation_gate(150, ENVIRONMENTS["lunar_lander"])


def test_feature_descriptors_rows_and_label_preservation():
    clean = np.random.default_rng(7).normal(size=(600, 8))
    injected, labels = inject_imposters(clean, 2, "gaussian", 7)
    descriptors = feature_descriptors(injected, labels, windows=12)
    n_features = injected.shape[1]
    assert len(descriptors) == 12 * n_features
    # Imposter columns (indices 8, 9) always carry label +1; originals carry -1.
    assert set(descriptors.loc[descriptors.feature == 8, "label"]) == {1}
    assert set(descriptors.loc[descriptors.feature == 0, "label"]) == {-1}


def test_deviation_gate_handles_both_environments():
    bipedal = ENVIRONMENTS["bipedal_walker"]
    assert deviation_gate(100, bipedal)  # above positive threshold
    assert deviation_gate(-5, bipedal)  # below negative threshold
    assert not deviation_gate(0, bipedal)  # inside the gate band


# --- models / evaluation --------------------------------------------------


def test_models_expose_expected_estimators():
    built = models(seed=7)
    assert set(built) == {"naive", "random_forest", "knn", "svm"}
    for estimator in built.values():
        assert hasattr(estimator, "fit") and hasattr(estimator, "predict")


def test_evaluate_records_and_predictions_schema(evaluation):
    records, predictions = evaluation
    assert records, "evaluate produced no records"
    record_keys = {"environment", "imposters", "noise", "metric", "model", "accuracy"}
    assert record_keys.issubset(records[0])
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in records)
    prediction_columns = {
        "environment", "imposters", "noise", "metric", "model", "true", "predicted"
    }
    assert prediction_columns.issubset(predictions.columns)
    assert set(predictions["true"].unique()).issubset({-1, 1})
    assert set(predictions["predicted"].unique()).issubset({-1, 1})


def test_benchmark_diagnostics_expose_reconstruction_gap(evaluation):
    records, _predictions = evaluation
    diagnostics = benchmark_diagnostics(records)
    assert set(diagnostics["environment"]) == set(ENVIRONMENTS)
    assert "ranking_agrees_with_published" in diagnostics
    assert diagnostics["local_accuracy_range"].gt(0).all()
    assert diagnostics["published_best_accuracy"].between(0, 1).all()
