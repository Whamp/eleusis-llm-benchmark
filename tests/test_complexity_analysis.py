"""Complexity analysis degrades gracefully without complexity data."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from eleusis.analysis.complexity import analyze_complexity


def test_analyze_complexity_skips_binning_when_metrics_absent(tmp_path: Path) -> None:
    """A frame whose rounds carry no rule complexity must not raise KeyError.

    Loader frames always include the complexity columns, but every value can
    be NaN when no rule metrics were computable (for example historical JSON
    without an embedded rules library). analyze_complexity must skip the
    misleading optimal-k report and binned stats while still writing the
    no-data outputs that scripts and the HTML report link to.
    """
    df = pd.DataFrame(
        {
            "model": ["m1", "m1"],
            "rule_description": ["rule a", "rule b"],
            "success": [True, False],
            "counting_success": [True, False],
            "score": [10.0, -2.0],
            "cyclomatic_complexity": [None, None],
            "node_count": [None, None],
        }
    )
    tee = io.StringIO()

    enriched = analyze_complexity(df, {"m1": "C0"}, tmp_path, tee)  # type: ignore[arg-type]

    assert isinstance(enriched, pd.DataFrame)
    text = tee.getvalue()
    assert "No rule complexity metrics" in text
    assert "Optimal K" not in text
    assert (tmp_path / "complexity_analysis.png").exists()
    assert (tmp_path / "complexity_analysis.json").exists()
