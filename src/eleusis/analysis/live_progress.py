"""Collect live Benchmark Run progress across worker result folders.

One source of truth for progress readers: the CLI progress command and the
web dashboard both consume :func:`collect_live_progress`. SQLite worker
folders expose per-Round rows derived from authoritative Round Records;
historical JSON-only folders expose only coarse checkpoint statistics.
"""

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from eleusis.benchmark_run_store import (
    BENCHMARK_RUN_DATABASE_NAME,
    BenchmarkRunStore,
    BenchmarkRunStoreError,
)


@dataclass(frozen=True)
class RoundProgress:
    """One scheduled Round's live status within a worker."""

    round_number: int
    rule_name: str
    status: str  # "completed" | "active" | "scheduled"
    score: int | None = None
    turn_count: int | None = None
    terminal_kind: str | None = None
    failed_guesses: int | None = None


@dataclass(frozen=True)
class WorkerProgress:
    """One worker folder's aggregated live progress."""

    name: str
    source: str  # "sqlite" | "legacy"
    completed: int = 0
    total: int = 0
    successful: int = 0
    score: int = 0
    duration_seconds: float = 0.0
    active_round_number: int | None = None
    committed_turns: int | None = None
    rounds: list[RoundProgress] = field(default_factory=list)
    usage: dict[str, object] | None = None
    error: str | None = None


def matching_worker_folders(pattern: str) -> list[Path]:
    """Find matching legacy-export and authoritative SQLite worker folders."""
    artifact_patterns = (
        f"results/{pattern}/results.json",
        f"results/{pattern}/{BENCHMARK_RUN_DATABASE_NAME}",
    )
    return sorted(
        {
            Path(artifact).parent
            for artifact_pattern in artifact_patterns
            for artifact in glob.glob(artifact_pattern)
        }
    )


def _sqlite_worker_progress(folder: Path) -> WorkerProgress:
    """Read live Round, Turn, score, and usage progress from SQLite."""
    store = BenchmarkRunStore(folder)
    manifest = store.read_manifest()
    schedule = cast(list[dict[str, object]], manifest["schedule"])
    progress = store.read_progress()
    summary = store.read_derived_summary()
    usage = cast(dict[str, object], summary["usage"])

    rounds: list[RoundProgress] = []
    for record in store.read_completed_rounds():
        values = store.derive_round_values(record)
        outcome = cast(dict[str, object], record["terminal_outcome"])
        rounds.append(
            RoundProgress(
                round_number=cast(int, values["round_number"]),
                rule_name=cast(
                    str, schedule[cast(int, values["round_number"]) - 1]["rule_name"]
                ),
                status="completed",
                score=cast(int, values["score"]),
                turn_count=cast(int, values["turn_count"]),
                terminal_kind=cast(str, outcome["kind"]),
                failed_guesses=cast(int, values["failed_guesses"]),
            )
        )

    active_round_number: int | None = None
    committed_turns: int | None = None
    if progress.active_round_number is not None:
        active_round_number = progress.active_round_number
        committed_turns = progress.committed_turns
        rounds.append(
            RoundProgress(
                round_number=active_round_number,
                rule_name=cast(str, schedule[active_round_number - 1]["rule_name"]),
                status="active",
                turn_count=committed_turns,
            )
        )
    for round_number in range(len(rounds) + 1, len(schedule) + 1):
        rounds.append(
            RoundProgress(
                round_number=round_number,
                rule_name=cast(str, schedule[round_number - 1]["rule_name"]),
                status="scheduled",
            )
        )

    return WorkerProgress(
        name=folder.name,
        source="sqlite",
        completed=progress.completed_rounds,
        total=progress.total_rounds,
        successful=cast(int, summary["successful_rounds"]),
        score=cast(int, summary["total_score"]),
        duration_seconds=cast(float, usage["duration_seconds"]),
        active_round_number=active_round_number,
        committed_turns=committed_turns,
        rounds=rounds,
        usage=usage,
    )


def _legacy_worker_progress(folder: Path) -> WorkerProgress:
    """Read one historical JSON-only worker progress projection."""
    with (folder / "results.json").open() as results_file:
        data = json.load(results_file)
    checkpoint = cast(dict[str, object], data["checkpoint"])
    statistics = cast(dict[str, object], data.get("statistics", {}))
    return WorkerProgress(
        name=folder.name,
        source="legacy",
        completed=cast(int, checkpoint["completed_rounds"]),
        total=cast(int, checkpoint["total_rounds"]),
        successful=cast(int, statistics.get("successful_rounds", 0)),
        score=cast(int, statistics.get("total_score", 0)),
        duration_seconds=cast(float, statistics.get("total_wall_clock_seconds", 0.0)),
    )


def collect_live_progress(pattern: str) -> list[WorkerProgress]:
    """Collect live progress for every worker folder matching the glob."""
    workers: list[WorkerProgress] = []
    for folder in matching_worker_folders(pattern):
        try:
            workers.append(
                _sqlite_worker_progress(folder)
                if (folder / BENCHMARK_RUN_DATABASE_NAME).is_file()
                else _legacy_worker_progress(folder)
            )
        except (
            BenchmarkRunStoreError,
            OSError,
            json.JSONDecodeError,
            KeyError,
        ) as error:
            workers.append(
                WorkerProgress(name=folder.name, source="sqlite", error=str(error))
            )
    return workers
