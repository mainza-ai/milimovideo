"""The M5 comparison tool must stay importable and cover every lane."""

from __future__ import annotations


def test_estimate_rows_cover_every_lane_and_match_the_registry():
    from benchmark_models import estimate_rows, markdown

    rows = estimate_rows()
    lanes = {row["lane"] for row in rows}
    assert lanes == {"video", "image", "sam", "depth"}
    for row in rows:
        assert row["estimated_peak_gb"] > 0
    video = next(row for row in rows if row["lane"] == "video")
    assert video["base_gb"] == 76.0
    table = markdown(rows)
    assert table.startswith("| Lane |")
    assert "86.8 GB" in table


def test_run_requires_explicit_confirmation(monkeypatch, capsys):
    import benchmark_models
    import pytest

    monkeypatch.setattr(
        "sys.argv",
        ["benchmark_models.py", "--run", "video", "--project", "p", "--out", "/tmp/x.json"],
    )
    with pytest.raises(SystemExit, match="loads a real model"):
        benchmark_models.main()
