"""Immutable scientific manifest for one Benchmark Run."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from eleusis.game.rule_library import RuleLibraryEntry
from eleusis.round_continuation import ROUND_CONTINUATION_VERSION

if TYPE_CHECKING:
    from eleusis.evaluation_startup import EvaluationStartup

BENCHMARK_RUN_DATABASE_VERSION = 1
BENCHMARK_RUN_MANIFEST_VERSION = 1
ROUND_RECORD_VERSION = 1
BENCHMARK_RUN_EXPORT_VERSION = 1


class _StrictManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _ManifestVersions(_StrictManifestModel):
    database: Literal[1]
    manifest: Literal[1]
    round_record: Literal[1]
    round_checkpoint: Literal[1]
    export: Literal[1]


class _ScheduledRound(_StrictManifestModel):
    round_number: int = Field(ge=1)
    rule_name: str | None
    rule_description: str | None
    rule_code: str | None
    batch_round_index: int = Field(ge=0)


class _FileProvenance(_StrictManifestModel):
    path: str
    sha256: str


class _SourceProvenance(_StrictManifestModel):
    revision: str
    dirty: bool
    files: list[_FileProvenance]
    fingerprint: str


class _PromptIdentity(_StrictManifestModel):
    name: str
    sha256: str


class _ModelIdentity(_StrictManifestModel):
    model_key: str
    display_name: str


class _CompilerIdentity(_StrictManifestModel):
    provider: str
    model_id: str
    display_name: str


class _EffectiveSettings(_StrictManifestModel):
    game_seed: int
    hand_size: int = Field(ge=1)
    max_turns: int = Field(ge=1)
    wrong_guess_penalty: int = Field(ge=0)
    shadow_mode: str
    llm_max_tokens: int = Field(ge=1)
    llm_temperature: float
    llm_seed: int | None
    llm_max_retries: int = Field(ge=1)
    rule_selection: str


class _BenchmarkRunManifest(_StrictManifestModel):
    version: Literal[1]
    run_id: str
    versions: _ManifestVersions
    schedule: list[_ScheduledRound]
    effective_settings: _EffectiveSettings
    model_identity: _ModelIdentity
    compiler_identity: _CompilerIdentity
    prompt_identities: list[_PromptIdentity]
    source_provenance: _SourceProvenance


def _sha256_bytes(content: bytes) -> str:
    """Return a lowercase SHA-256 digest for scientific identity data."""
    return hashlib.sha256(content).hexdigest()


def _repository_root() -> Path:
    """Return the repository root containing the installed source module."""
    return Path(__file__).resolve().parents[2]


def _git_output(repository: Path, *arguments: str) -> str:
    """Run one read-only Git query used by source provenance."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Benchmark Run source provenance unavailable: git {' '.join(arguments)}"
        ) from error
    return result.stdout


def capture_source_provenance() -> dict[str, object]:
    """Fingerprint current behavior-affecting tracked and untracked source files."""
    repository = _repository_root()
    revision = _git_output(repository, "rev-parse", "HEAD").strip()
    status = _git_output(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    listed_paths = _git_output(
        repository,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    ).splitlines()
    root_files = {
        "config.yaml",
        "models.yaml",
        "pyproject.toml",
        "rules.json",
        "suites.yaml",
    }
    behavior_paths = sorted(
        path
        for path in listed_paths
        if path.startswith(("src/", "scripts/")) or path in root_files
    )
    files = [
        {"path": path, "sha256": _sha256_bytes((repository / path).read_bytes())}
        for path in behavior_paths
        if (repository / path).is_file()
    ]
    fingerprint_input = "\n".join(
        [revision, *(f"{item['path']}\0{item['sha256']}" for item in files)]
    ).encode()
    return {
        "revision": revision,
        "dirty": bool(status),
        "files": files,
        "fingerprint": _sha256_bytes(fingerprint_input),
    }


def capture_prompt_identities() -> list[dict[str, str]]:
    """Identify exact prompt source content fixed for a Benchmark Run."""
    prompt_directory = Path(__file__).resolve().parent / "prompts"
    return [
        {"name": path.stem, "sha256": _sha256_bytes(path.read_bytes())}
        for path in sorted(prompt_directory.glob("*.py"))
        if path.name != "__init__.py"
    ]


def _schedule_rule(
    startup: EvaluationStartup,
    rules: list[RuleLibraryEntry],
    round_number: int,
) -> tuple[RuleLibraryEntry | None, int]:
    """Resolve one immutable scheduled rule and batch index when deterministic."""
    if startup.suite_cases:
        rule_name, batch_index = startup.suite_cases[round_number - 1]
        rule = next((item for item in rules if item.get("name") == rule_name), None)
        return rule, batch_index
    offset = startup.game_config.get("batch_round_offset")
    batch_index = (
        offset
        if offset is not None
        else (round_number - 1) % startup.num_rounds_per_rule
    )
    if startup.rules_config["selection"] != "sequential":
        return None, batch_index
    rule_offset = (round_number - 1) // startup.num_rounds_per_rule
    rule_index = (startup.rules_config["index"] + rule_offset) % len(rules)
    return rules[rule_index], batch_index


def _build_round_schedule(
    startup: EvaluationStartup,
    rules: list[RuleLibraryEntry],
) -> list[dict[str, object]]:
    """Build the immutable ordered Round schedule fixed at Run creation."""
    schedule: list[dict[str, object]] = []
    for round_number in range(1, startup.num_rounds + 1):
        rule, batch_index = _schedule_rule(startup, rules, round_number)
        schedule.append(
            {
                "round_number": round_number,
                "rule_name": rule.get("name") if rule else None,
                "rule_description": rule["description"] if rule else None,
                "rule_code": rule["code"] if rule else None,
                "batch_round_index": batch_index,
            }
        )
    return schedule


def create_benchmark_run_manifest(
    startup: EvaluationStartup,
    rules: list[RuleLibraryEntry],
    *,
    run_id: str,
) -> dict[str, object]:
    """Create and strictly validate one immutable Benchmark Run manifest."""
    game = startup.game_config
    game_seed = game["seed"]
    if game_seed is None:
        raise ValueError(
            "Benchmark Run manifest requires a concrete effective game seed"
        )
    compiler = startup.config["rule_compiler"]
    llm = startup.config["llm"]
    payload = {
        "version": BENCHMARK_RUN_MANIFEST_VERSION,
        "run_id": run_id,
        "versions": {
            "database": BENCHMARK_RUN_DATABASE_VERSION,
            "manifest": BENCHMARK_RUN_MANIFEST_VERSION,
            "round_record": ROUND_RECORD_VERSION,
            "round_checkpoint": ROUND_CONTINUATION_VERSION,
            "export": BENCHMARK_RUN_EXPORT_VERSION,
        },
        "schedule": _build_round_schedule(startup, rules),
        "effective_settings": {
            "game_seed": game_seed,
            "hand_size": game["hand_size"],
            "max_turns": game["max_turns"],
            "wrong_guess_penalty": game["wrong_guess_penalty"],
            "shadow_mode": game.get("shadow_mode", "offline"),
            "llm_max_tokens": llm["max_tokens"],
            "llm_temperature": llm["temperature"],
            "llm_seed": llm["seed"],
            "llm_max_retries": llm["max_llm_retries"],
            "rule_selection": startup.rules_config["selection"],
        },
        "model_identity": {
            "model_key": startup.player_model,
            "display_name": startup.player_display_name,
        },
        "compiler_identity": {
            "provider": compiler["provider"],
            "model_id": compiler["model_id"],
            "display_name": startup.rule_compiler_display_name,
        },
        "prompt_identities": capture_prompt_identities(),
        "source_provenance": capture_source_provenance(),
    }
    return cast(
        dict[str, object],
        _BenchmarkRunManifest.model_validate(payload).model_dump(mode="json"),
    )
