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
