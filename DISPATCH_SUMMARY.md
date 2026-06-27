# Dispatch Summary — adversarial-rl-feature-selection

**Round:** 2 (deeper edge-case tests, validation, research-backed diagnostics)
**Branch:** `feature/claude-local-enhancements` (local only; no push)
**Date:** 2026-06-27

## Files changed

- `src/robustfeatures/core.py` — added three standalone helpers (`sturges_bins`,
  `freedman_diaconis_bins`, `majority_class_baseline`) and a `windows` validation
  guard in `feature_descriptors`.
- `tests/test_diagnostics.py` — new file; 25 tests cross-checking the helpers
  against independent references (NumPy bin estimators, scikit-learn
  DummyClassifier) and exercising the new guard's error paths.
- `README.md` — documented the new diagnostic helpers and the `windows` guard.

All source additions are **additive and behavior-preserving**: every new helper
is deliberately NOT wired into `evaluate`, exactly like the round-1
`mutual_information` helper, so the reconstructed metrics in `reports/latest/`
are unchanged. This directly implements the round-1 DEFERRED items #3, #4, #5.

## Research candidates

Domain: information-theoretic detection of injected ("imposter") noise features
in RL observation profiles, via entropy / joint-entropy / KL descriptors +
classifiers.

**DONE this round**

1. **Adaptive histogram binning** (round-1 deferred #3). Added `sturges_bins(n)`
   (`ceil(log2(n)) + 1`, Sturges 1926) and `freedman_diaconis_bins(values)`
   (`h = 2·IQR / n^(1/3)`, Freedman–Diaconis 1981; IQR-based so robust to the
   heavy tails of injected noise, unlike Scott's rule). Both return integer bin
   counts and were verified this session to match NumPy's own
   `histogram_bin_edges(..., bins="sturges"|"fd")` across multiple sample sizes;
   FD falls back to one bin on degenerate (constant) input. Standalone, so the
   fixed `bins=20`/`bins=12` defaults in `entropy`/`joint_entropy`/`kl_divergence`
   are untouched and reproduced metrics are unchanged.
2. **Majority-class accuracy floor** (round-1 deferred #5). Added
   `majority_class_baseline(labels)`, the "most_frequent"/zero-rule baseline
   (`DummyClassifier(strategy="most_frequent")`), cross-checked this session
   against scikit-learn's DummyClassifier to machine precision and a hand value
   (`[1,1,1,-1] -> 0.75`). On the 8-original/2-imposter descriptor labels the
   floor is exactly `0.8`, giving honest context for the detector accuracies.
   Standalone; not wired into `evaluate`.
3. **`feature_descriptors` window guard** (round-1 deferred #4). `windows < 1`
   or `windows > n_samples` now raises a clear `ValueError`. Previously
   `windows > n_samples` produced empty `np.array_split` windows that crashed
   `kl_divergence` with an opaque "zero-size array to reduction operation
   minimum" error (reproduced this session on a 5-row input before the fix).
   The normal path (`samples >> windows`, e.g. the default `windows=12` over
   900/3600 rows) never triggers the guard, so behavior is identical there.

**DEFERRED** (would change reproduced outputs, or out of scope this round)

- Wire MI / an `mrmr`-style relevance−redundancy score into the descriptor
  matrix as an additional detector input — changes `metrics.json`; needs a fresh
  honest full run.
- Use `freedman_diaconis_bins` to drive an opt-in adaptive-binning mode in
  `entropy`/`evaluate` — changes numeric outputs; needs a deliberate full
  reproduction.
- Report `majority_class_baseline` as a row in `summary.csv`/`benchmark_diagnostics.csv`
  — changes a committed artifact schema; needs a fresh full run.
- Per-feature (rather than per-label-class) imposter-vs-original separability
  diagnostic.

Source URLs (ideas only; re-implemented independently):
- https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
- https://en.wikipedia.org/wiki/Sturges%27s_rule
- https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
- https://arxiv.org/pdf/1408.1487 (information-theoretic feature selection, round-1 thread)

## Hygiene / robustness wins

- `feature_descriptors` hardened against the `windows > n_samples` zero-size
  reduction (clear `ValueError` with remediation hint).
- ruff and mypy remained clean throughout (no new findings introduced by the
  additions); mypy still type-checks all source files cleanly.
- Test coverage: **22 → 47** tests. The 25 new tests pin each helper to an
  independent reference rather than to its own output, and lock the guard's
  error paths and the `windows == n_samples` boundary.

## Exact local test / lint / type output

```
$ python3 -m ruff check . --exclude .venv
All checks passed!

$ MYPYPATH=src python3 -m mypy --cache-dir=/tmp/arfs_mypy
Success: no issues found in 4 source files

$ PYTHONPATH=src MPLBACKEND=Agg python3 -m pytest -q
...............................................                          [100%]
47 passed in ~32s
```

Honest end-to-end check: `MPLBACKEND=Agg PYTHONPATH=src python3 -m robustfeatures.run --smoke`
completed (exit 0) and wrote all expected artifacts to a fresh (gitignored)
`artifacts/*-smoke/` dir; `benchmark_diagnostics.csv` still reports
`ranking_agrees_with_published=False` for both environments (reconstruction gap
preserved). `reports/latest/` was not modified (smoke does not publish). No
headline metrics in README were altered.

## Environment note

Ran tests/lint/type with a clean Linux interpreter (the committed `.venv` is a
macOS build); Python 3.10 was used for execution while the source stays
3.12-targeted. Deps installed via `pip --break-system-packages`; mypy/pytest
caches kept under `/tmp`. The mount blocks `unlink(2)`, so git index/ref locks
were swept aside and the index was staged via `GIT_INDEX_FILE` under `/tmp`;
"unable to unlink" warnings during git writes are benign. Pre-existing untracked
macOS duplicate files (`reports/latest/*\ 2.*`) were left untouched and NOT
staged.
