import json
import platform

import numpy as np
import pandas as pd

from robustfeatures.artifacts import environment, output_dir, publish_latest, save
from robustfeatures.core import ENVIRONMENTS, benchmark_diagnostics, evaluate
from robustfeatures.run import diagnostics_markdown


def test_save_roundtrips_numpy_and_drops_non_finite(tmp_path):
    target = tmp_path / "payload.json"
    save(
        target,
        {
            "float": np.float64(1.5),
            "int": np.int64(3),
            "array": np.array([1, 2, 3]),
            "nan": float("nan"),
            "inf": np.float64("inf"),
            "nested": {"tuple": (1, 2)},
        },
    )
    loaded = json.loads(target.read_text())
    assert loaded["float"] == 1.5
    assert loaded["int"] == 3
    assert loaded["array"] == [1, 2, 3]
    assert loaded["nan"] is None  # non-finite values are nulled, never NaN tokens
    assert loaded["inf"] is None
    assert loaded["nested"]["tuple"] == [1, 2]


def test_environment_reports_runtime_metadata():
    meta = environment()
    assert set(meta) == {"python", "platform", "generated_at_utc"}
    assert meta["python"] == platform.python_version()


def test_output_dir_creates_distinct_smoke_and_full_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    smoke = output_dir(smoke=True)
    full = output_dir(smoke=False)
    assert smoke.exists() and smoke.name.endswith("-smoke")
    assert full.exists() and full.name.endswith("-full")


def test_publish_latest_copies_files_and_records_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "artifacts" / "run"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text("{}\n")
    publish_latest(out)
    assert (tmp_path / "reports" / "latest" / "metrics.json").exists()
    source = (tmp_path / "reports" / "SOURCE_RUN.txt").read_text()
    assert "run" in source


def test_diagnostics_markdown_renders_table_per_environment():
    records, _predictions = evaluate(samples=360, seed=7)
    diagnostics = benchmark_diagnostics(records)
    markdown = diagnostics_markdown(diagnostics)
    assert markdown.startswith("| Environment |")
    for name in ENVIRONMENTS:
        assert name in markdown
    # header (2 lines) + one row per environment
    assert len(markdown.splitlines()) == 2 + len(ENVIRONMENTS)


def test_diagnostics_markdown_accepts_empty_frame():
    empty = pd.DataFrame(
        columns=[
            "environment",
            "local_best_model",
            "local_best_metric",
            "local_best_accuracy",
            "published_best_model",
            "published_best_metric",
            "published_best_accuracy",
            "ranking_agrees_with_published",
            "local_accuracy_range",
        ]
    )
    markdown = diagnostics_markdown(empty)
    assert len(markdown.splitlines()) == 2  # header only
