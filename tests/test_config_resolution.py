"""Contracts for resolving the benchmark configuration path.

C1 Relative config names resolve against the repository root, not the
   caller's working directory, so the documented command
   `uv run python scripts/evaluate_single.py --config config.smoke.yaml`
   works from any CWD and parallel workers stay CWD-independent.
C2 Absolute config paths load unchanged.
C3 A missing config path raises a clear file-not-found error.
"""

import os
from pathlib import Path

import pytest

from eleusis.evaluation_support import load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_relative_config_resolves_against_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C1: a bare config filename loads even when the CWD is elsewhere."""
    monkeypatch.chdir(tmp_path)
    assert os.getcwd() == str(tmp_path)

    config = load_config("config.smoke.yaml")

    assert config["game"]["max_turns"] == 8
    assert config["llm"]["max_tokens"] > 0


def test_absolute_config_path_loads() -> None:
    """C2: absolute paths are used as-is."""
    config = load_config(str(_REPO_ROOT / "config.smoke.yaml"))

    assert config["game"]["max_turns"] == 8


def test_missing_config_raises_filenotfound() -> None:
    """C3: a missing config surfaces the underlying file error."""
    with pytest.raises(FileNotFoundError, match=r"no\_such\_config\.yaml"):
        load_config("no_such_config.yaml")
