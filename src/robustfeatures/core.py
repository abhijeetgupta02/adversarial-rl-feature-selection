from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    dimensions: int
    imposter_counts: tuple[int, ...]
    positive_return_threshold: float
    negative_return_threshold: float
    training_protocol: str


ENVIRONMENTS = {
    "lunar_lander": EnvironmentSpec(
        "LunarLander-v3", 8, (1, 2, 3, 4), 300.0, 100.0, "PPO, 200,000 steps"
    ),
    "bipedal_walker": EnvironmentSpec(
        "BipedalWalker-v3", 24, (1, 2, 4, 8), 96.0, -1.97, "ARS, 1,000 iterations"
    ),
}

PUBLISHED_RESULTS = {
    "lunar_lander": {
        "naive": {"entropy": 0.9400, "joint_entropy": 0.8054, "kl": 0.6032},
        "random_forest": {"entropy": 0.9705, "joint_entropy": 0.9506, "kl": 0.7962},
        "knn": {"entropy": 0.9411, "joint_entropy": 0.9135, "kl": 0.7654},
        "svm": {"entropy": 0.9111, "joint_entropy": 0.7592, "kl": 0.6913},
    },
    "bipedal_walker": {
        "naive": {"entropy": 0.9500, "joint_entropy": 0.8856, "kl": 0.8345},
        "random_forest": {"entropy": 0.9333, "joint_entropy": 0.9250, "kl": 0.9078},
        "knn": {"entropy": 0.9555, "joint_entropy": 0.9600, "kl": 0.9555},
        "svm": {"entropy": 0.9777, "joint_entropy": 0.9577, "kl": 0.9577},
    },
}


def entropy(values: np.ndarray, bins: int = 20) -> float:
    counts, _ = np.histogram(values, bins=bins)
    probability = counts[counts > 0] / counts.sum()
    return float(-(probability * np.log(probability)).sum())


def joint_entropy(left: np.ndarray, right: np.ndarray, bins: int = 12) -> float:
    counts, _, _ = np.histogram2d(left, right, bins=bins)
    probability = counts[counts > 0] / counts.sum()
    return float(-(probability * np.log(probability)).sum())


def kl_divergence(left: np.ndarray, right: np.ndarray, bins: int = 20) -> float:
    low = min(left.min(), right.min())
    high = max(left.max(), right.max())
    edges = np.linspace(low, high + 1e-9, bins + 1)
    p, _ = np.histogram(left, bins=edges)
    q, _ = np.histogram(right, bins=edges)
    p = (p + 1e-8) / (p.sum() + bins * 1e-8)
    q = (q + 1e-8) / (q.sum() + bins * 1e-8)
    return float(np.sum(p * np.log(p / q)))


def mutual_information(left: np.ndarray, right: np.ndarray, bins: int = 12) -> float:
    """Plug-in mutual information I(X;Y) from a shared 2D histogram.

    Equivalent to H(X) + H(Y) - H(X, Y) when all terms use the joint histogram's
    consistent marginals; computed as the KL divergence between the joint and the
    product of marginals, so the estimate is always non-negative. In the
    feature-relevance/redundancy framing this measures how much information a
    feature shares with another, complementing the entropy/joint-entropy/KL
    descriptors. See information-theoretic feature-selection literature
    (e.g. https://arxiv.org/pdf/1408.1487).
    """
    counts, _, _ = np.histogram2d(left, right, bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    joint = counts / total
    outer = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    mask = (joint > 0) & (outer > 0)
    return float(np.sum(joint[mask] * np.log(joint[mask] / outer[mask])))


def sturges_bins(n: int) -> int:
    """Sturges' rule for histogram bin count: ``ceil(log2(n)) + 1``.

    Sturges (1926) assumes an approximately normal sample and tends to
    under-bin large or skewed samples, but it is a cheap, parameter-free
    default. Matches ``numpy.histogram_bin_edges(x, bins="sturges")``.

    Provided as a standalone descriptor-tuning helper; it is intentionally
    NOT wired into :func:`entropy`/:func:`evaluate`, so the reconstructed
    metrics in ``reports/latest/`` are unchanged. See
    https://en.wikipedia.org/wiki/Sturges%27s_rule and the NumPy histogram
    bin-estimator docs.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return int(np.ceil(np.log2(n))) + 1


def freedman_diaconis_bins(values: np.ndarray) -> int:
    """Freedman-Diaconis histogram bin count from the interquartile range.

    Bin width ``h = 2 * IQR(x) / n**(1/3)`` (Freedman & Diaconis, 1981);
    the count is ``ceil((max - min) / h)``. Using the IQR instead of the
    standard deviation makes it robust to the heavy tails / outliers common
    in injected-imposter feature distributions, unlike Scott's rule. On
    degenerate input (zero IQR or zero range, e.g. a constant feature) it
    falls back to a single bin. Matches ``numpy.histogram_bin_edges(x,
    bins="fd")`` on non-degenerate data.

    Standalone helper, intentionally NOT wired into the default
    entropy/KL descriptors, so reconstructed metrics are unchanged. See
    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule.
    """
    flat = np.asarray(values, dtype=float).ravel()
    if flat.size < 1:
        raise ValueError("values must be non-empty")
    quartiles = np.percentile(flat, [75, 25])
    iqr = float(quartiles[0] - quartiles[1])
    span = float(flat.max() - flat.min())
    width = 2.0 * iqr / (flat.size ** (1.0 / 3.0))
    if width <= 0.0 or span <= 0.0:
        return 1
    return int(np.ceil(span / width))


def synthetic_observations(spec: EnvironmentSpec, samples: int, seed: int) -> np.ndarray:
    """Deterministic trajectory fixture preserving each paper environment's dimensionality."""
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(spec.dimensions, spec.dimensions))
    latent = rng.normal(size=(samples, spec.dimensions))
    observations = latent @ mixing / np.sqrt(spec.dimensions)
    observations += 0.15 * np.sin(np.linspace(0, 10, samples))[:, None]
    return observations


def inject_imposters(
    observations: np.ndarray,
    count: int,
    noise: str,
    seed: int,
    in_range: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if noise == "gaussian":
        imposters = rng.normal(0, 1, (len(observations), count))
    elif noise == "uniform":
        bound = 1.0 if in_range else 6.0
        imposters = rng.uniform(-bound, bound, (len(observations), count))
    else:
        raise ValueError(f"Unsupported noise: {noise}")
    return np.column_stack([observations, imposters]), np.r_[
        -np.ones(observations.shape[1], dtype=int), np.ones(count, dtype=int)
    ]


def feature_descriptors(
    observations: np.ndarray, labels: np.ndarray, windows: int = 12
) -> pd.DataFrame:
    if windows < 1:
        raise ValueError(f"windows must be >= 1, got {windows}")
    n_samples = observations.shape[0]
    if n_samples < windows:
        raise ValueError(
            f"windows ({windows}) exceeds the number of observation rows "
            f"({n_samples}); each window must hold at least one row or the "
            "entropy/KL descriptors hit a zero-size reduction. Use "
            "windows <= n_samples."
        )
    rows = []
    for window in np.array_split(observations, windows):
        feature_entropies = np.array([entropy(window[:, index]) for index in range(window.shape[1])])
        mean_entropy = feature_entropies.mean()
        for index in range(window.shape[1]):
            others = [other for other in range(window.shape[1]) if other != index]
            joint = np.mean([joint_entropy(window[:, index], window[:, other]) for other in others])
            divergence = np.mean(
                [kl_divergence(window[:, index], window[:, other]) for other in others]
            )
            rows.append(
                {
                    "feature": index,
                    "mean_entropy": mean_entropy,
                    "entropy": feature_entropies[index],
                    "centered_entropy": feature_entropies[index] - mean_entropy,
                    "joint_entropy": joint,
                    "kl": divergence,
                    "label": labels[index],
                }
            )
    return pd.DataFrame(rows)


def deviation_gate(delta_return: float, spec: EnvironmentSpec) -> bool:
    return bool(
        delta_return > spec.positive_return_threshold
        or delta_return < spec.negative_return_threshold
    )


def models(seed: int = 7):
    return {
        "naive": LinearRegression(),
        "random_forest": RandomForestClassifier(n_estimators=500, random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(kernel="rbf", C=10, gamma="scale"),
    }


def majority_class_baseline(labels: np.ndarray) -> float:
    """Accuracy floor of always predicting the most frequent label.

    This is the "most_frequent"/"zero rule" baseline (equivalent to
    ``sklearn.dummy.DummyClassifier(strategy="most_frequent")``) and gives
    the trivial accuracy any detector must beat. With the descriptor matrix's
    -1 (original) vs +1 (imposter) labels and many more originals than
    imposters, this floor can be high, so reporting it alongside the four
    detector families contextualizes their accuracy. Deterministic and
    standalone: NOT wired into :func:`evaluate`, so reconstructed metrics are
    unchanged. See
    https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    flat = np.asarray(labels).ravel()
    if flat.size == 0:
        raise ValueError("labels must be non-empty")
    _, counts = np.unique(flat, return_counts=True)
    return float(counts.max()) / float(flat.size)


def evaluate(samples: int = 3600, seed: int = 7) -> tuple[list[dict], pd.DataFrame]:
    records: list[dict] = []
    prediction_rows: list[dict] = []
    metric_columns = {
        "entropy": ["mean_entropy", "centered_entropy", "entropy"],
        "joint_entropy": ["mean_entropy", "centered_entropy", "joint_entropy"],
        "kl": ["mean_entropy", "centered_entropy", "kl"],
    }
    for environment_name, spec in ENVIRONMENTS.items():
        clean = synthetic_observations(spec, samples, seed)
        for count in spec.imposter_counts:
            for noise in ("gaussian", "uniform"):
                injected, labels = inject_imposters(clean, count, noise, seed + count)
                descriptors = feature_descriptors(injected, labels)
                train = descriptors.index % 3 != 0
                test = ~train
                for metric, columns in metric_columns.items():
                    for model_name, model in models(seed).items():
                        model.fit(descriptors.loc[train, columns], descriptors.loc[train, "label"])
                        raw = model.predict(descriptors.loc[test, columns])
                        predicted = np.where(raw >= 0, 1, -1)
                        accuracy = accuracy_score(descriptors.loc[test, "label"], predicted)
                        records.append(
                            {
                                "environment": environment_name,
                                "imposters": count,
                                "noise": noise,
                                "metric": metric,
                                "model": model_name,
                                "accuracy": accuracy,
                            }
                        )
                        prediction_rows.extend(
                            {
                                "environment": environment_name,
                                "imposters": count,
                                "noise": noise,
                                "metric": metric,
                                "model": model_name,
                                "true": int(truth),
                                "predicted": int(prediction),
                            }
                            for truth, prediction in zip(
                                descriptors.loc[test, "label"], predicted, strict=True
                            )
                        )
    return records, pd.DataFrame(prediction_rows)


def benchmark_diagnostics(records: list[dict]) -> pd.DataFrame:
    """Compare local fixture ordering against the published best rows."""
    frame = pd.DataFrame(records)
    summary = (
        frame.groupby(["environment", "model", "metric"], as_index=False)
        .accuracy.mean()
        .sort_values(["environment", "accuracy"], ascending=[True, False])
    )
    rows = []
    for environment, env_summary in summary.groupby("environment", sort=True):
        local_best = env_summary.iloc[0]
        published_best_model, published_best_metric, published_best_accuracy = max(
            (
                (model, metric, accuracy)
                for model, metrics in PUBLISHED_RESULTS[environment].items()
                for metric, accuracy in metrics.items()
            ),
            key=lambda row: row[2],
        )
        rows.append(
            {
                "environment": environment,
                "local_best_model": local_best["model"],
                "local_best_metric": local_best["metric"],
                "local_best_accuracy": float(local_best["accuracy"]),
                "published_best_model": published_best_model,
                "published_best_metric": published_best_metric,
                "published_best_accuracy": float(published_best_accuracy),
                "ranking_agrees_with_published": bool(
                    local_best["model"] == published_best_model
                    and local_best["metric"] == published_best_metric
                ),
                "local_accuracy_range": float(
                    env_summary["accuracy"].max() - env_summary["accuracy"].min()
                ),
            }
        )
    return pd.DataFrame(rows)
