# Dispatch Summary — adversarial-rl-feature-selection

**Round:** 1 (hygiene + baseline tests + doc fixes)
**Branch:** `feature/claude-local-enhancements` (local only; no push)
**Date:** 2026-06-25

## Files changed

- `src/robustfeatures/core.py` — typing fixes; added `mutual_information` descriptor.
- `pyproject.toml` — pinned `[tool.mypy]` config; added `mypy` dev dependency.
- `tests/test_core.py` — expanded from 4 to 16 tests (edge cases, error paths, new descriptor).
- `tests/test_artifacts.py` — new file; 6 tests covering the artifact writers and markdown renderer.
- `Makefile` — added `lint` and `typecheck` targets.
- `.github/workflows/ci.yml` — added a `uv run mypy` step.
- `README.md` — documented dev commands and the new descriptor.

## Research candidates

Domain: information-theoretic detection of injected ("imposter") noise features in
RL observation profiles, via entropy / joint-entropy / KL descriptors + classifiers.

**DONE**

1. **Mutual-information descriptor** (`mutual_information(x, y)`). Standalone, non-negative
   plug-in estimate of `I(X;Y)` computed as the KL divergence between the joint histogram
   and the product of its marginals. Fits the relevance/redundancy framing from
   information-theoretic feature selection. Implemented as a tested helper and **deliberately
   not wired into `evaluate`**, so the reconstructed metrics in `reports/latest/` are unchanged
   (no behavior change, no fabricated results).

**DEFERRED** (out of R1 scope / would change reproduced outputs)

2. Wire MI (and an `mrmr`-style relevance−redundancy score) into the descriptor matrix as an
   additional detector input — changes `metrics.json`, so deferred to a deeper round with a
   fresh honest full run.
3. Adaptive histogram binning (Freedman–Diaconis / Sturges) for the entropy estimators —
   changes numeric outputs; defer.
4. Input validation in `feature_descriptors` for `windows > n_samples` (empty-window guard so
   `kl_divergence` cannot hit a zero-size reduction on pathological inputs).
5. A constant/majority-class baseline detector to contextualize accuracy.

Source URLs (ideas only; re-implemented independently):
- https://arxiv.org/pdf/1408.1487 (robust feature selection via mutual-information distributions)
- https://www.mdpi.com/journal/entropy/special_issues/feature_selection_big_data
- https://thuijskens.github.io/2017/10/07/feature-selection/

## Hygiene / robustness wins

- mypy: **4 errors → 0**. Fixed missing annotations on `evaluate`'s `records`/`prediction_rows`,
  added `-> pd.DataFrame` to `feature_descriptors`, and refactored the `published_best`
  selection in `benchmark_diagnostics` to typed tuples (removed two `object`-typed errors).
  Behavior verified identical via the smoke run's `benchmark_diagnostics.csv`.
- Pinned mypy config + dev dep and added a CI mypy gate so the typed state is enforced.
- Test coverage: 4 → 22 tests, now exercising error paths (unsupported noise), the
  in-range/out-of-range uniform branch, fixture determinism, descriptor label preservation,
  the artifact JSON writer (numpy + non-finite handling), and the markdown renderer.

## Exact local test / lint / type output

```
$ python3 -m ruff check . --exclude .venv
All checks passed!

$ python3 -m mypy
Success: no issues found in 4 source files

$ PYTHONPATH=src python3 -m pytest -q
......................                                                   [100%]
22 passed in 32.88s
```

Honest end-to-end check: `MPLBACKEND=Agg PYTHONPATH=src python3 -m robustfeatures.run --smoke`
completed (exit 0) and wrote all expected artifacts; `benchmark_diagnostics.csv` shows
`ranking_agrees_with_published=False` for both environments (reconstruction gap preserved).
No headline metrics in README were altered.

## Environment note

Ran tests/lint/type with a clean Linux interpreter (the committed `.venv` is a macOS build).
Deps installed via `pip --break-system-packages`; mypy/pytest caches kept under `/tmp`.
