"""Scientific manifest provenance and compatibility contracts."""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from eleusis import benchmark_run_manifest, runner
from eleusis.benchmark_config import ModelConfig
from eleusis.benchmark_run_manifest import (
    BenchmarkRunManifestIncompatibilityError,
    create_benchmark_run_manifest,
    restore_benchmark_run_config,
    verify_benchmark_run_resume_compatibility,
)
from eleusis.game.rule_library import RuleLibraryEntry
from tests.conftest import FakeLLMClient
from tests.test_evaluation_orchestrator import _startup


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
    lockfile = repository / "uv.lock"
    lockfile.write_text("version = 1\n")
    (repository / ".gitignore").write_text("logs/\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fixture")
    monkeypatch.setattr(benchmark_run_manifest, "_repository_root", lambda: repository)

    clean = benchmark_run_manifest.capture_source_provenance()
    assert clean["dirty"] is False

    lockfile.write_text("version = 2\n")
    dirty_lock = benchmark_run_manifest.capture_source_provenance()
    assert dirty_lock["dirty"] is True
    assert dirty_lock["fingerprint"] != clean["fingerprint"]
    lockfile.write_text("version = 1\n")

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


def test_dirty_uv_lock_produces_field_specific_resume_incompatibility(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A changed uv lockfile fails resume through source provenance identity."""
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    (repository / "uv.lock").write_text("version = 1\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fixture")
    monkeypatch.setattr(benchmark_run_manifest, "_repository_root", lambda: repository)
    startup = _startup()
    rules: list[RuleLibraryEntry] = [
        {
            "name": "only_red",
            "description": "Only red cards.",
            "code": "return card.color == 'red'",
        }
    ]
    manifest = create_benchmark_run_manifest(
        startup,
        rules,
        run_id="lock-resume",
        configured_game_seed=7,
    )
    (repository / "uv.lock").write_text("version = 2\n")

    with pytest.raises(
        BenchmarkRunManifestIncompatibilityError,
        match=r"source_provenance\.dirty changed",
    ):
        verify_benchmark_run_resume_compatibility(manifest, startup.config)


def test_restore_config_merges_only_matching_backup_provider_api_keys(
    monkeypatch: MonkeyPatch,
) -> None:
    """Credential-free manifests recover matching fallback credentials at resume."""
    startup = _startup()
    backup: ModelConfig = {
        "provider": "openai_compat",
        "model_id": "backup-model",
        "base_url": "https://backup.example/v1",
        "api_key": "SECRET-BACKUP-KEY",
    }
    startup.config["rule_compiler"]["backup_providers"] = [backup]
    manifest = create_benchmark_run_manifest(
        startup,
        [
            {
                "name": "only_red",
                "description": "Only red cards.",
                "code": "return card.color == 'red'",
            }
        ],
        run_id="backup-credential-resume",
        configured_game_seed=7,
    )
    assert "SECRET-BACKUP-KEY" not in str(manifest)
    current = copy.deepcopy(startup.config)

    restored = restore_benchmark_run_config(manifest, current)

    restored_backups = restored["rule_compiler"]["backup_providers"]
    assert restored_backups[0]["api_key"] == "SECRET-BACKUP-KEY"
    seen_configs: list[ModelConfig] = []

    def create_compiler_client(
        config: ModelConfig,
        **_kwargs: object,
    ) -> FakeLLMClient:
        seen_configs.append(config)
        return FakeLLMClient()

    monkeypatch.setattr(runner, "create_client_from_config", create_compiler_client)
    monkeypatch.setattr(
        runner,
        "create_client",
        lambda *_args, **_kwargs: FakeLLMClient(),
    )
    compiler, _scientist = runner._create_round_clients(restored, "Scientist")

    assert len(compiler.fallback_clients) == 1
    assert seen_configs[1]["api_key"] == "SECRET-BACKUP-KEY"
