"""Scientific manifest provenance and compatibility contracts."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pytest import MonkeyPatch

from eleusis import benchmark_run_manifest


def _git(repository: Path, *arguments: str) -> None:
    """Run one successful Git setup command for a provenance fixture."""
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def test_source_provenance_fingerprints_all_behavior_input_states(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Tracked, staged, modified, and permitted untracked inputs affect identity."""
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    (repository / "src").mkdir()
    tracked = repository / "src" / "behavior.py"
    tracked.write_text("VALUE = 1\n")
    (repository / ".gitignore").write_text("logs/\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fixture")
    monkeypatch.setattr(benchmark_run_manifest, "_repository_root", lambda: repository)

    clean = benchmark_run_manifest.capture_source_provenance()
    assert clean["dirty"] is False

    tracked.write_text("VALUE = 2\n")
    modified = benchmark_run_manifest.capture_source_provenance()
    assert modified["dirty"] is True
    assert modified["fingerprint"] != clean["fingerprint"]

    _git(repository, "add", str(tracked))
    staged = benchmark_run_manifest.capture_source_provenance()
    assert staged["dirty"] is True
    assert staged["fingerprint"] == modified["fingerprint"]

    tracked.write_text("VALUE = 3\n")
    staged_and_modified = benchmark_run_manifest.capture_source_provenance()
    assert staged_and_modified["fingerprint"] != staged["fingerprint"]

    (repository / "scripts").mkdir()
    (repository / "scripts" / "new_behavior.py").write_text("VALUE = 4\n")
    with_untracked = benchmark_run_manifest.capture_source_provenance()
    assert with_untracked["fingerprint"] != staged_and_modified["fingerprint"]

    (repository / "logs").mkdir()
    (repository / "logs" / "ignored.log").write_text("not scientific\n")
    with_ignored_output = benchmark_run_manifest.capture_source_provenance()
    assert with_ignored_output == with_untracked
