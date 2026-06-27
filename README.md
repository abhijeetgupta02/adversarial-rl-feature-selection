# Imposter Injection Feature Detection

Paper-faithful public reconstruction of *Imposter Injection: Learning to Select
Features in Reinforcement Learning* (KSE 2024).

## Best Evidence

- **What I built:** a public reconstruction of imposter-feature detection for
  Lunar Lander and Bipedal Walker dimensional profiles.
- **What is reproduced:** Gaussian/uniform imposter injection, entropy,
  joint-entropy, KL descriptors, return-deviation gates, and four detector
  families.
- **What is reconstructed:** original PPO/ARS policies and trajectories were
  unavailable, so executable runs use deterministic trajectory fixtures.
- **Main verified result:** local fixture best rows reach `0.9831` accuracy on
  Lunar Lander and `0.9429` on Bipedal Walker; diagnostics explicitly show the
  synthetic ranking does not reproduce the published best ordering.
- **How to verify:** `uv sync --frozen && make test && make reproduce-smoke`.

The repository implements appended Gaussian/uniform imposters, Lunar Lander
and Bipedal Walker dimensional profiles, entropy/joint-entropy/KL feature
descriptors, the paper's return-deviation gates, and Naive/RF/KNN/SVM
detectors. Original trained PPO/ARS policies and trajectories were unavailable,
so local experiments use deterministic trajectory fixtures.

Full runs also emit `benchmark_diagnostics.csv` and `BENCHMARK_NOTE.md`, which
compare the local fixture's best model/metric ordering with the published best
rows. This makes the reconstruction gap inspectable instead of implying that
synthetic rankings are paper results.

```bash
uv sync
make test
make reproduce-smoke
make reproduce-results
```

## Development

```bash
make test       # uv run pytest
make lint       # uv run ruff check .
make typecheck  # uv run mypy
```

`tests/` covers the information measures, imposter injection (including the
unsupported-noise error path and in-range vs out-of-range uniform support), the
deterministic fixtures, descriptor construction, and the artifact writers.

`robustfeatures.core.mutual_information(x, y)` is also available as an
information-theoretic descriptor. It returns a non-negative plug-in estimate of
`I(X;Y)` (the KL divergence between the joint histogram and the product of its
marginals) for relevance/redundancy framing. It is provided as a standalone
helper and is intentionally not wired into `evaluate`, so the reconstructed
metrics in `reports/latest/` are unchanged.

### Diagnostic helpers

Three further standalone helpers support descriptor tuning and accuracy
context. They are likewise **not** wired into `evaluate`, so the metrics in
`reports/latest/` are unchanged:

- `sturges_bins(n)` and `freedman_diaconis_bins(values)` estimate a histogram
  bin count via Sturges' rule (`ceil(log2(n)) + 1`) and the
  outlier-robust Freedman–Diaconis rule (`h = 2·IQR / n^(1/3)`). Both match
  NumPy's own `histogram_bin_edges(..., bins="sturges"|"fd")` and let you probe
  whether the fixed `bins=20`/`bins=12` defaults under- or over-bin a given
  feature.
- `majority_class_baseline(labels)` returns the trivial "always predict the
  most frequent label" accuracy floor (the
  `DummyClassifier(strategy="most_frequent")` baseline) that any detector must
  beat.

`feature_descriptors` now validates `windows`: a value below `1` or above the
number of observation rows raises a clear `ValueError` instead of failing later
with an opaque zero-size reduction.
