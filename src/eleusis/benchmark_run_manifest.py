"""Immutable scientific manifest for one Benchmark Run."""

from __future__ import annotations

import copy
import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from eleusis.benchmark_config import BenchmarkConfig, parse_benchmark_config
from eleusis.game.rule_library import RuleLibraryEntry
from eleusis.round_continuation import ROUND_CONTINUATION_VERSION

if TYPE_CHECKING:
    from eleusis.evaluation_startup import EvaluationStartup

BENCHMARK_RUN_DATABASE_VERSION = 1
BENCHMARK_RUN_MANIFEST_VERSION = 1
ROUND_RECORD_VERSION = 1
BENCHMARK_RUN_EXPORT_VERSION = 1


class BenchmarkRunManifestIncompatibilityError(ValueError):
    """Raised when current scientific inputs cannot resume a Benchmark Run."""


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
    configured_game_seed: int | None
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
    scientific_config: dict[str, JsonValue]
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


def _is_behavior_path(path: str) -> bool:
    """Return whether one repository path can affect benchmark behavior."""
    return path.startswith(("src/", "scripts/")) or path in {
        "config.yaml",
        "models.yaml",
        "pyproject.toml",
        "rules.json",
        "suites.yaml",
    }


def capture_source_provenance() -> dict[str, object]:
    """Fingerprint current behavior-affecting tracked and untracked source files."""
    repository = _repository_root()
    revision = _git_output(repository, "rev-parse", "HEAD").strip()
    status_lines = _git_output(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    status_paths = {
        line[3:].split(" -> ")[-1] for line in status_lines if len(line) >= 4
    }
    listed_paths = _git_output(
        repository,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    ).splitlines()
    behavior_paths = sorted(path for path in listed_paths if _is_behavior_path(path))
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
        "dirty": any(_is_behavior_path(path) for path in status_paths),
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


def _without_credentials(value: object) -> JsonValue:
    """Copy JSON configuration while excluding secret credential fields."""
    if isinstance(value, Mapping):
        return {
            str(key): _without_credentials(item)
            for key, item in value.items()
            if key != "api_key"
        }
    if isinstance(value, list):
        return [_without_credentials(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "Benchmark Run scientific configuration contains a non-JSON value: "
        f"{type(value).__name__}"
    )


def create_benchmark_scientific_config(
    config: BenchmarkConfig,
    *,
    configured_game_seed: int | None,
) -> dict[str, JsonValue]:
    """Project fixed scientific settings while excluding operational options."""
    game = config["game"]
    scientific_game: dict[str, JsonValue] = {
        "num_rules": game.get("num_rules"),
        "num_rounds_per_rule": game.get("num_rounds_per_rule"),
        "max_turns": game["max_turns"],
        "hand_size": game["hand_size"],
        "wrong_guess_penalty": game["wrong_guess_penalty"],
        "seed": configured_game_seed,
        "batch_round_offset": game.get("batch_round_offset"),
        "shadow_mode": game.get("shadow_mode", "offline"),
    }
    return {
        "model": config.get("model"),
        "game": scientific_game,
        "llm": cast(JsonValue, _without_credentials(config["llm"])),
        "rule_compiler": cast(JsonValue, _without_credentials(config["rule_compiler"])),
        "rules": {
            "selection": config["rules"]["selection"],
            "index": config["rules"]["index"],
        },
        "suite": config.get("suite"),
    }


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


def validate_benchmark_run_manifest_document(payload: object) -> dict[str, object]:
    """Strictly decode one supported Benchmark Run manifest document."""
    if not isinstance(payload, Mapping):
        raise BenchmarkRunManifestIncompatibilityError(
            "Benchmark Run manifest incompatible: document must be an object"
        )
    version = payload.get("version")
    if version != BENCHMARK_RUN_MANIFEST_VERSION:
        raise BenchmarkRunManifestIncompatibilityError(
            "Benchmark Run manifest incompatible: unsupported version "
            f"{version!r}; expected {BENCHMARK_RUN_MANIFEST_VERSION}"
        )
    try:
        document = _BenchmarkRunManifest.model_validate(payload)
    except ValidationError as error:
        raise BenchmarkRunManifestIncompatibilityError(
            f"Benchmark Run manifest incompatible: malformed version {version}: {error}"
        ) from error
    return cast(dict[str, object], document.model_dump(mode="json"))


def _first_manifest_difference(
    stored: object,
    current: object,
    path: str,
) -> str | None:
    """Return the first stable dotted field path whose scientific value changed."""
    if isinstance(stored, Mapping) and isinstance(current, Mapping):
        for key in sorted(set(stored) | set(current)):
            child_path = f"{path}.{key}"
            if key not in stored or key not in current:
                return child_path
            difference = _first_manifest_difference(
                stored[key], current[key], child_path
            )
            if difference is not None:
                return difference
        return None
    if stored != current:
        return path
    return None


def verify_benchmark_run_resume_compatibility(
    manifest: Mapping[str, object],
    current_config: BenchmarkConfig,
) -> None:
    """Fail closed when current scientific inputs differ from the stored manifest."""
    validated = validate_benchmark_run_manifest_document(manifest)
    effective_settings = cast(Mapping[str, object], validated["effective_settings"])
    current_scientific = create_benchmark_scientific_config(
        current_config,
        configured_game_seed=current_config["game"]["seed"],
    )
    difference = _first_manifest_difference(
        validated["scientific_config"],
        current_scientific,
        "scientific_config",
    )
    if difference is None:
        current_prompts = capture_prompt_identities()
        difference = _first_manifest_difference(
            validated["prompt_identities"],
            current_prompts,
            "prompt_identities",
        )
    if difference is None:
        current_source = capture_source_provenance()
        stored_source = cast(Mapping[str, object], validated["source_provenance"])
        for field in ("revision", "dirty", "fingerprint"):
            if stored_source[field] != current_source[field]:
                difference = f"source_provenance.{field}"
                break
    if difference is not None:
        raise BenchmarkRunManifestIncompatibilityError(
            f"Benchmark Run resume incompatible: {difference} changed"
        )
    if effective_settings["game_seed"] is None:
        raise BenchmarkRunManifestIncompatibilityError(
            "Benchmark Run resume incompatible: effective_settings.game_seed missing"
        )


def restore_benchmark_run_config(
    manifest: Mapping[str, object],
    current_config: BenchmarkConfig,
) -> BenchmarkConfig:
    """Rebuild effective runtime config from fixed scientific and operational inputs."""
    validated = validate_benchmark_run_manifest_document(manifest)
    scientific = copy.deepcopy(cast(dict[str, object], validated["scientific_config"]))
    game = cast(dict[str, object], scientific["game"])
    effective = cast(Mapping[str, object], validated["effective_settings"])
    game["seed"] = effective["game_seed"]
    if "pause_after_turn" in current_config["game"]:
        game["pause_after_turn"] = current_config["game"]["pause_after_turn"]
    rules = cast(dict[str, object], scientific["rules"])
    rules["library_path"] = current_config["rules"]["library_path"]
    compiler = cast(dict[str, object], scientific["rule_compiler"])
    if "api_key" in current_config["rule_compiler"]:
        compiler["api_key"] = current_config["rule_compiler"]["api_key"]
    if scientific.get("suite") is None:
        scientific.pop("suite", None)
    return parse_benchmark_config(scientific)


def create_benchmark_run_manifest(
    startup: EvaluationStartup,
    rules: list[RuleLibraryEntry],
    *,
    run_id: str,
    configured_game_seed: int | None,
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
            "configured_game_seed": configured_game_seed,
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
        "scientific_config": create_benchmark_scientific_config(
            startup.config,
            configured_game_seed=configured_game_seed,
        ),
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
    return validate_benchmark_run_manifest_document(payload)
