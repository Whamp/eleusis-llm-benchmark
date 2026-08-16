"""Authoritative per-Run SQLite persistence and portable JSON export."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eleusis.benchmark_run_manifest import (
    BENCHMARK_RUN_DATABASE_VERSION,
    BENCHMARK_RUN_EXPORT_VERSION,
    ROUND_RECORD_VERSION,
    validate_benchmark_run_manifest_document,
)
from eleusis.evaluation_results import TurnRecord
from eleusis.round_continuation import (
    ROUND_CONTINUATION_VERSION,
    capture_round_continuation,
    validate_round_continuation_document,
)
from eleusis.round_record import (
    append_round_record_turn,
    complete_round_record,
    create_active_round_record,
    validate_round_record_document,
)
from eleusis.shadow_verdict import (
    SHADOW_VERDICT_VERSION,
    validate_shadow_verdict_document,
)

if TYPE_CHECKING:
    from eleusis.round_execution import RoundRuntime

BENCHMARK_RUN_DATABASE_NAME = "benchmark_run.sqlite3"


class BenchmarkRunStoreError(RuntimeError):
    """Raised when the authoritative Benchmark Run store rejects an operation."""


@dataclass(frozen=True)
class ActiveStoredRound:
    """Validated active Round Record and its hidden continuation checkpoint."""

    round_number: int
    record: dict[str, object]
    continuation: dict[str, object]


@dataclass(frozen=True)
class BenchmarkRunProgress:
    """Live schedule cursor derived from authoritative Round rows."""

    run_id: str
    total_rounds: int
    completed_rounds: int
    active_round_number: int | None
    committed_turns: int
    next_round_number: int | None
    is_complete: bool


@dataclass(frozen=True)
class BenchmarkRunExportStatus:
    """Compare a portable JSON export with SQLite's terminal-data watermark."""

    authoritative_watermark: int
    export_watermark: int | None
    export_version: int | None
    is_current: bool


class BenchmarkRunStore:
    """Own SQLite transactions and portable exports for one Benchmark Run."""

    def __init__(self, run_folder: Path) -> None:
        """Open an existing per-Run store rooted in its portable Run folder."""
        self.run_folder = run_folder.resolve()
        self.database_path = self.run_folder / BENCHMARK_RUN_DATABASE_NAME
        self.export_path = self.run_folder / "results.json"
        if not self.database_path.is_file():
            raise BenchmarkRunStoreError(
                f"Benchmark Run store unavailable: {self.database_path} does not exist"
            )
        self._verify_database_version()

    @classmethod
    def create(
        cls,
        run_folder: Path,
        manifest: Mapping[str, object],
    ) -> BenchmarkRunStore:
        """Create one authoritative SQLite database and immutable Run manifest."""
        run_folder.mkdir(parents=True, exist_ok=True)
        database_path = run_folder / BENCHMARK_RUN_DATABASE_NAME
        if database_path.exists():
            raise BenchmarkRunStoreError(
                f"Benchmark Run store already exists: {database_path}"
            )
        connection = sqlite3.connect(database_path)
        try:
            cls._configure_connection(connection)
            cls._create_schema(connection)
            with connection:
                connection.execute(
                    """
                    INSERT INTO benchmark_run (
                        run_id,
                        database_version,
                        manifest_version,
                        manifest_document,
                        transaction_sequence,
                        completed_sequence,
                        export_sequence
                    ) VALUES (?, ?, ?, ?, 0, 0, 0)
                    """,
                    (
                        manifest["run_id"],
                        BENCHMARK_RUN_DATABASE_VERSION,
                        manifest["version"],
                        cls._encode_document(manifest),
                    ),
                )
        finally:
            connection.close()
        return cls(run_folder)

    @staticmethod
    def _configure_connection(connection: sqlite3.Connection) -> None:
        """Apply durable SQLite settings to one short-lived connection."""
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        """Create the version-one document-oriented per-Run schema."""
        connection.executescript(
            """
            CREATE TABLE benchmark_run (
                run_id TEXT PRIMARY KEY,
                database_version INTEGER NOT NULL,
                manifest_version INTEGER NOT NULL,
                manifest_document TEXT NOT NULL,
                transaction_sequence INTEGER NOT NULL,
                completed_sequence INTEGER NOT NULL,
                export_sequence INTEGER NOT NULL
            ) STRICT;

            CREATE TABLE rounds (
                run_id TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('active', 'completed')),
                record_version INTEGER NOT NULL,
                checkpoint_version INTEGER,
                record_document TEXT NOT NULL,
                continuation_document TEXT,
                transaction_sequence INTEGER NOT NULL,
                PRIMARY KEY (run_id, round_number),
                FOREIGN KEY (run_id) REFERENCES benchmark_run(run_id),
                CHECK (
                    (status = 'active' AND checkpoint_version IS NOT NULL
                        AND continuation_document IS NOT NULL)
                    OR
                    (status = 'completed' AND checkpoint_version IS NULL
                        AND continuation_document IS NULL)
                )
            ) STRICT;

            CREATE TRIGGER reject_completed_round_update
            BEFORE UPDATE ON rounds
            WHEN OLD.status = 'completed'
            BEGIN
                SELECT RAISE(ABORT, 'immutable completed Round Record update rejected');
            END;

            CREATE TRIGGER reject_completed_round_delete
            BEFORE DELETE ON rounds
            WHEN OLD.status = 'completed'
            BEGIN
                SELECT RAISE(ABORT, 'immutable completed Round Record delete rejected');
            END;

            CREATE TABLE shadow_verdicts (
                run_id TEXT NOT NULL,
                verdict_id TEXT NOT NULL,
                proposal_id TEXT NOT NULL,
                verdict_version INTEGER NOT NULL,
                verdict_document TEXT NOT NULL,
                transaction_sequence INTEGER NOT NULL,
                PRIMARY KEY (run_id, verdict_id),
                FOREIGN KEY (run_id) REFERENCES benchmark_run(run_id)
            ) STRICT;

            CREATE TRIGGER reject_shadow_verdict_update
            BEFORE UPDATE ON shadow_verdicts
            BEGIN
                SELECT RAISE(ABORT, 'immutable Shadow Verdict update rejected');
            END;

            CREATE TRIGGER reject_shadow_verdict_delete
            BEFORE DELETE ON shadow_verdicts
            BEGIN
                SELECT RAISE(ABORT, 'immutable Shadow Verdict delete rejected');
            END;
            """
        )

    def _connect(self) -> sqlite3.Connection:
        """Open one configured connection with mapping-style rows."""
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        self._configure_connection(connection)
        return connection

    def _verify_database_version(self) -> None:
        """Reject databases outside this store implementation's exact schema."""
        try:
            with closing(sqlite3.connect(self.database_path)) as connection:
                row = connection.execute(
                    "SELECT database_version FROM benchmark_run"
                ).fetchone()
        except sqlite3.Error as error:
            raise BenchmarkRunStoreError(
                "Benchmark Run database incompatible: "
                f"{self.database_path} has a schema defect: {error}"
            ) from error
        if row is None or row[0] != BENCHMARK_RUN_DATABASE_VERSION:
            found = None if row is None else row[0]
            raise BenchmarkRunStoreError(
                "Benchmark Run database incompatible: "
                f"found version {found!r}, expected {BENCHMARK_RUN_DATABASE_VERSION}"
            )

    @staticmethod
    def _encode_document(payload: Mapping[str, object]) -> str:
        """Encode one validated domain document deterministically."""
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _decode_document(payload: str) -> object:
        """Decode one stored JSON document with a store-specific diagnostic."""
        try:
            return json.loads(payload)
        except json.JSONDecodeError as error:
            raise BenchmarkRunStoreError(
                f"Benchmark Run store document is invalid JSON: {error}"
            ) from error

    @staticmethod
    def _next_transaction_sequence(connection: sqlite3.Connection) -> int:
        """Advance and return the global transaction ordering sequence."""
        row = connection.execute(
            """
            UPDATE benchmark_run
            SET transaction_sequence = transaction_sequence + 1
            RETURNING transaction_sequence
            """
        ).fetchone()
        if row is None:
            raise BenchmarkRunStoreError(
                "Benchmark Run store invariant broken: manifest row is missing"
            )
        return cast(int, row[0])

    def read_manifest(self) -> dict[str, object]:
        """Read the immutable Benchmark Run manifest document."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT manifest_document FROM benchmark_run"
            ).fetchone()
        if row is None:
            raise BenchmarkRunStoreError(
                "Benchmark Run store invariant broken: manifest row is missing"
            )
        document = self._decode_document(cast(str, row["manifest_document"]))
        return validate_benchmark_run_manifest_document(document)

    def start_round(
        self,
        runtime: RoundRuntime,
        *,
        effective_round_seed: int | None,
        batch_round_index: int,
    ) -> None:
        """Atomically commit initial Round setup before the first Model Attempt."""
        if effective_round_seed is None:
            raise BenchmarkRunStoreError(
                "Benchmark Run checkpoint requires a concrete effective Round seed"
            )
        manifest = self.read_manifest()
        record = create_active_round_record(
            manifest,
            runtime,
            effective_round_seed=effective_round_seed,
            batch_round_index=batch_round_index,
        )
        continuation = capture_round_continuation(
            runtime,
            [],
            next_turn_index=0,
        )
        with closing(self._connect()) as connection, connection:
            sequence = self._next_transaction_sequence(connection)
            try:
                connection.execute(
                    """
                    INSERT INTO rounds (
                        run_id,
                        round_number,
                        status,
                        record_version,
                        checkpoint_version,
                        record_document,
                        continuation_document,
                        transaction_sequence
                    ) VALUES (?, ?, 'active', ?, ?, ?, ?, ?)
                    """,
                    (
                        manifest["run_id"],
                        runtime.round_number,
                        record["version"],
                        continuation["version"],
                        self._encode_document(record),
                        self._encode_document(continuation),
                        sequence,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise BenchmarkRunStoreError(
                    "Benchmark Run initial checkpoint rejected: "
                    f"Round {runtime.round_number} already exists"
                ) from error

    @staticmethod
    def _validate_active_round_versions(row: sqlite3.Row) -> None:
        """Fail closed on SQL versions before decoding active domain documents."""
        record_version = cast(int, row["record_version"])
        if record_version != ROUND_RECORD_VERSION:
            raise BenchmarkRunStoreError(
                "Benchmark Run active Round Record incompatible: "
                f"found version {record_version}, expected {ROUND_RECORD_VERSION}"
            )
        checkpoint_version = cast(int, row["checkpoint_version"])
        if checkpoint_version != ROUND_CONTINUATION_VERSION:
            raise BenchmarkRunStoreError(
                "Benchmark Run active checkpoint incompatible: "
                f"found version {checkpoint_version}, "
                f"expected {ROUND_CONTINUATION_VERSION}"
            )

    def _decode_active_round(self, row: sqlite3.Row) -> ActiveStoredRound:
        """Validate one active SQL row and both versioned domain documents."""
        self._validate_active_round_versions(row)
        round_number = cast(int, row["round_number"])
        record = validate_round_record_document(
            self._decode_document(cast(str, row["record_document"]))
        )
        continuation = validate_round_continuation_document(
            self._decode_document(cast(str, row["continuation_document"]))
        )
        continuation_round = cast(Mapping[str, object], continuation["round"])
        if not (
            round_number
            == record["scheduled_round_number"]
            == continuation_round["number"]
        ):
            raise BenchmarkRunStoreError(
                "Benchmark Run active Round identity incompatible: SQL Round, "
                "Round Record, and continuation numbers disagree"
            )
        return ActiveStoredRound(
            round_number=round_number,
            record=record,
            continuation=continuation,
        )

    def read_active_round(self, round_number: int) -> ActiveStoredRound | None:
        """Read and validate one active Round Record and continuation checkpoint."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT round_number, record_version, checkpoint_version,
                       record_document, continuation_document
                FROM rounds
                WHERE round_number = ? AND status = 'active'
                """,
                (round_number,),
            ).fetchone()
        return None if row is None else self._decode_active_round(row)

    def read_resumable_round(self) -> ActiveStoredRound | None:
        """Read the sole active Round available for Benchmark Run resume."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT round_number, record_version, checkpoint_version,
                       record_document, continuation_document
                FROM rounds
                WHERE status = 'active'
                ORDER BY round_number
                """
            ).fetchall()
        if len(rows) > 1:
            raise BenchmarkRunStoreError(
                "Benchmark Run resume invariant broken: multiple active Rounds"
            )
        return None if not rows else self._decode_active_round(rows[0])

    def read_progress(self) -> BenchmarkRunProgress:
        """Read and validate live completed-Round and committed-Turn progress."""
        manifest = self.read_manifest()
        schedule = cast(list[Mapping[str, object]], manifest["schedule"])
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT round_number, status, record_version, checkpoint_version,
                       record_document, continuation_document
                FROM rounds
                ORDER BY round_number
                """
            ).fetchall()
        completed_numbers = [
            cast(int, row["round_number"])
            for row in rows
            if row["status"] == "completed"
        ]
        expected_completed = list(range(1, len(completed_numbers) + 1))
        if completed_numbers != expected_completed:
            raise BenchmarkRunStoreError(
                "Benchmark Run progress invariant broken: completed Rounds are not "
                "a contiguous schedule prefix"
            )
        active_rows = [row for row in rows if row["status"] == "active"]
        if len(active_rows) > 1:
            raise BenchmarkRunStoreError(
                "Benchmark Run progress invariant broken: multiple active Rounds"
            )
        active = self._decode_active_round(active_rows[0]) if active_rows else None
        if active is not None and active.round_number != len(completed_numbers) + 1:
            raise BenchmarkRunStoreError(
                "Benchmark Run progress invariant broken: active Round does not "
                "follow completed schedule prefix"
            )
        if len(completed_numbers) > len(schedule) or (
            active is not None and active.round_number > len(schedule)
        ):
            raise BenchmarkRunStoreError(
                "Benchmark Run progress invariant broken: Round exceeds stored schedule"
            )
        committed_turns = 0
        if active is not None:
            next_turn_index = active.continuation["next_turn_index"]
            if not isinstance(next_turn_index, int):
                raise BenchmarkRunStoreError(
                    "Benchmark Run progress invariant broken: committed Turn cursor "
                    "is malformed"
                )
            committed_turns = next_turn_index
        is_complete = len(completed_numbers) == len(schedule) and active is None
        next_round_number = None
        if not is_complete:
            next_round_number = (
                active.round_number
                if active is not None
                else len(completed_numbers) + 1
            )
        return BenchmarkRunProgress(
            run_id=cast(str, manifest["run_id"]),
            total_rounds=len(schedule),
            completed_rounds=len(completed_numbers),
            active_round_number=(active.round_number if active is not None else None),
            committed_turns=committed_turns,
            next_round_number=next_round_number,
            is_complete=is_complete,
        )

    def commit_completed_turn(
        self,
        runtime: RoundRuntime,
        turn_records: list[TurnRecord],
    ) -> None:
        """Validate and atomically replace the checkpoint after a complete Turn."""
        active = self.read_active_round(runtime.round_number)
        if active is None:
            raise BenchmarkRunStoreError(
                "Benchmark Run Turn checkpoint rejected: "
                f"Round {runtime.round_number} is not active"
            )
        if not turn_records:
            raise BenchmarkRunStoreError(
                "Benchmark Run Turn checkpoint rejected: no completed Turn supplied"
            )
        continuation = capture_round_continuation(
            runtime,
            turn_records,
            next_turn_index=len(turn_records),
        )
        record = append_round_record_turn(
            active.record,
            active.continuation,
            continuation,
            turn_records[-1],
            runtime,
        )
        with closing(self._connect()) as connection, connection:
            sequence = self._next_transaction_sequence(connection)
            cursor = connection.execute(
                """
                UPDATE rounds
                SET record_document = ?,
                    continuation_document = ?,
                    transaction_sequence = ?
                WHERE round_number = ? AND status = 'active'
                """,
                (
                    self._encode_document(record),
                    self._encode_document(continuation),
                    sequence,
                    runtime.round_number,
                ),
            )
            if cursor.rowcount != 1:
                raise BenchmarkRunStoreError(
                    "Benchmark Run Turn checkpoint rejected: active Round changed"
                )

    def complete_round(self, round_number: int, *, game_over_reason: str) -> None:
        """Atomically promote one terminal Round Record and remove continuation."""
        active = self.read_active_round(round_number)
        if active is None:
            raise BenchmarkRunStoreError(
                "Benchmark Run terminal promotion rejected: "
                f"Round {round_number} is not active"
            )
        completed = complete_round_record(
            active.record,
            game_over_reason=game_over_reason,
        )
        with closing(self._connect()) as connection, connection:
            sequence = self._next_transaction_sequence(connection)
            cursor = connection.execute(
                """
                UPDATE rounds
                SET status = 'completed',
                    record_document = ?,
                    checkpoint_version = NULL,
                    continuation_document = NULL,
                    transaction_sequence = ?
                WHERE round_number = ? AND status = 'active'
                """,
                (self._encode_document(completed), sequence, round_number),
            )
            if cursor.rowcount != 1:
                raise BenchmarkRunStoreError(
                    "Benchmark Run terminal promotion rejected: active Round changed"
                )
            connection.execute(
                "UPDATE benchmark_run SET completed_sequence = ?",
                (sequence,),
            )
        self.ensure_current_export()

    def read_completed_round(self, round_number: int) -> dict[str, object]:
        """Read and validate one immutable completed Round Record."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT record_document
                FROM rounds
                WHERE round_number = ? AND status = 'completed'
                """,
                (round_number,),
            ).fetchone()
        if row is None:
            raise BenchmarkRunStoreError(
                f"Benchmark Run completed Round unavailable: {round_number}"
            )
        return validate_round_record_document(
            self._decode_document(cast(str, row["record_document"]))
        )

    def _completed_rounds(self) -> list[dict[str, object]]:
        """Read all immutable Round Records in scheduled order."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT record_document
                FROM rounds
                WHERE status = 'completed'
                ORDER BY round_number
                """
            ).fetchall()
        return [
            validate_round_record_document(
                self._decode_document(cast(str, row["record_document"]))
            )
            for row in rows
        ]

    def read_completed_rounds(self) -> list[dict[str, object]]:
        """Read all immutable completed Round Records in scheduled order."""
        return self._completed_rounds()

    @staticmethod
    def _round_shadow_proposal_ids(record: Mapping[str, object]) -> set[str]:
        """Collect immutable Shadow Guess proposal identities from one Round."""
        proposal_ids: set[str] = set()
        turns = cast(list[Mapping[str, object]], record["turns"])
        for turn in turns:
            guess = turn.get("guess_attempt")
            if isinstance(guess, Mapping) and guess.get("kind") == "shadow":
                proposal_id = guess.get("proposal_id")
                if isinstance(proposal_id, str):
                    proposal_ids.add(proposal_id)
        return proposal_ids

    def add_shadow_verdict(self, payload: Mapping[str, object]) -> None:
        """Append one immutable offline Shadow Verdict for a completed proposal."""
        verdict = validate_shadow_verdict_document(payload)
        proposal_ids = {
            proposal_id
            for record in self._completed_rounds()
            for proposal_id in self._round_shadow_proposal_ids(record)
        }
        proposal_id = cast(str, verdict["proposal_id"])
        if proposal_id not in proposal_ids:
            raise BenchmarkRunStoreError(
                "Benchmark Run Shadow Verdict rejected: completed proposal "
                f"{proposal_id!r} is unavailable"
            )
        manifest = self.read_manifest()
        with closing(self._connect()) as connection, connection:
            sequence = self._next_transaction_sequence(connection)
            try:
                connection.execute(
                    """
                    INSERT INTO shadow_verdicts (
                        run_id,
                        verdict_id,
                        proposal_id,
                        verdict_version,
                        verdict_document,
                        transaction_sequence
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        manifest["run_id"],
                        verdict["verdict_id"],
                        proposal_id,
                        verdict["version"],
                        self._encode_document(verdict),
                        sequence,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise BenchmarkRunStoreError(
                    "Benchmark Run Shadow Verdict rejected: immutable verdict "
                    f"{verdict['verdict_id']!r} already exists"
                ) from error
            connection.execute(
                "UPDATE benchmark_run SET completed_sequence = ?",
                (sequence,),
            )
        self.ensure_current_export()

    def read_shadow_verdicts(self) -> list[dict[str, object]]:
        """Read immutable offline Shadow Verdicts in append order."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT verdict_version, verdict_document
                FROM shadow_verdicts
                ORDER BY transaction_sequence
                """
            ).fetchall()
        verdicts: list[dict[str, object]] = []
        for row in rows:
            version = cast(int, row["verdict_version"])
            if version != SHADOW_VERDICT_VERSION:
                raise BenchmarkRunStoreError(
                    "Benchmark Run Shadow Verdict incompatible: "
                    f"found version {version}, expected {SHADOW_VERDICT_VERSION}"
                )
            verdicts.append(
                validate_shadow_verdict_document(
                    self._decode_document(cast(str, row["verdict_document"]))
                )
            )
        return verdicts

    def _export_sequences(self) -> tuple[int, int]:
        """Return terminal-data and generated-export ordering watermarks."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT completed_sequence, export_sequence FROM benchmark_run"
            ).fetchone()
        if row is None:
            raise BenchmarkRunStoreError(
                "Benchmark Run store invariant broken: manifest row is missing"
            )
        return cast(int, row["completed_sequence"]), cast(int, row["export_sequence"])

    def read_export_status(self) -> BenchmarkRunExportStatus:
        """Identify a missing, stale, or current export without rewriting it."""
        completed_sequence, export_sequence = self._export_sequences()
        export_watermark: int | None = None
        export_version: int | None = None
        try:
            document = json.loads(self.export_path.read_text())
        except (OSError, json.JSONDecodeError):
            document = None
        if isinstance(document, dict):
            version_value = document.get("version")
            watermark_value = document.get("watermark")
            if isinstance(version_value, int):
                export_version = version_value
            if isinstance(watermark_value, int):
                export_watermark = watermark_value
        return BenchmarkRunExportStatus(
            authoritative_watermark=completed_sequence,
            export_watermark=export_watermark,
            export_version=export_version,
            is_current=(
                export_sequence == completed_sequence
                and export_version == BENCHMARK_RUN_EXPORT_VERSION
                and export_watermark == completed_sequence
            ),
        )

    def _export_is_current(self, completed_sequence: int) -> bool:
        """Check whether the portable export has the authoritative watermark."""
        status = self.read_export_status()
        return (
            status.authoritative_watermark == completed_sequence and status.is_current
        )

    @staticmethod
    def _derived_model_usage(record: Mapping[str, object]) -> dict[str, object]:
        """Derive aggregate usage and throughput from immutable Model Attempts."""
        turns = cast(list[Mapping[str, object]], record["turns"])
        attempts = [
            attempt
            for turn in turns
            for attempt in cast(list[Mapping[str, object]], turn["model_attempts"])
        ]
        token_names = (
            "prompt_tokens",
            "output_tokens",
            "reasoning_tokens",
            "answer_tokens",
        )
        token_totals = {
            name: sum(
                cast(
                    int,
                    cast(Mapping[str, object], attempt["token_metrics"])[name],
                )
                for attempt in attempts
            )
            for name in token_names
        }
        provider_calls = [
            provider_call
            for attempt in attempts
            for provider_call in cast(
                list[Mapping[str, object]],
                attempt["provider_calls"],
            )
        ]
        duration_seconds = sum(
            cast(float, provider_call["duration_seconds"])
            for provider_call in provider_calls
        )
        output_tokens = token_totals["output_tokens"]
        return {
            "model_attempt_count": len(attempts),
            "provider_call_count": len(provider_calls),
            **token_totals,
            "duration_seconds": round(duration_seconds, 6),
            "throughput_tokens_per_second": round(
                output_tokens / duration_seconds if duration_seconds else 0.0,
                2,
            ),
            "cost": {
                "usd": None,
                "pricing_version": None,
            },
        }

    @staticmethod
    def _turn_has_correct_guess_observation(turn: Mapping[str, object]) -> bool:
        """Return whether a Formal or observed online Shadow Guess was correct."""
        guess = turn.get("guess_attempt")
        if not isinstance(guess, Mapping):
            return False
        if guess.get("correct") is True:
            return True
        online_evaluation = guess.get("online_evaluation")
        return (
            isinstance(online_evaluation, Mapping)
            and online_evaluation.get("correct") is True
        )

    @staticmethod
    def derive_round_values(record: Mapping[str, object]) -> dict[str, object]:
        """Recompute report values from authoritative terminal Round facts."""
        turns = cast(list[Mapping[str, object]], record["turns"])
        outcome = cast(Mapping[str, object], record["terminal_outcome"])
        settings = cast(Mapping[str, object], record["settings"])
        failed_guesses: list[object] = []
        if turns:
            last_turn = turns[-1]
            post_state = cast(Mapping[str, object], last_turn["post_card_state"])
            failed_guesses = cast(list[object], post_state["failed_rule_guesses"])
        max_turns = cast(int, settings["max_turns"])
        penalty = cast(int, settings["wrong_guess_penalty"]) * len(failed_guesses)
        score = -penalty
        if outcome["kind"] == "correct_formal_guess":
            score = max_turns - len(turns) + 1 - penalty
        correct_turns = [
            cast(int, turn["turn_number"])
            for turn in turns
            if BenchmarkRunStore._turn_has_correct_guess_observation(turn)
        ]
        first_correct_turn = min(correct_turns) if correct_turns else None
        no_stakes_score = (
            max_turns - first_correct_turn + 1 if first_correct_turn is not None else 0
        )
        return {
            "round_number": cast(int, record["scheduled_round_number"]),
            "turn_count": len(turns),
            "score": score,
            "floored_score": max(0, score),
            "no_stakes_score": no_stakes_score,
            "first_correct_turn": first_correct_turn,
            "failed_guesses": len(failed_guesses),
            "schema_compliance_rate": (
                sum(1 for turn in turns if not turn["schema_errors"]) / len(turns)
                if turns
                else None
            ),
            "usage": BenchmarkRunStore._derived_model_usage(record),
        }

    @staticmethod
    def _derived_run_summary(
        records: list[dict[str, object]],
        rounds: list[dict[str, object]],
    ) -> dict[str, object]:
        """Derive aggregate progress, outcomes, scores, and usage from Round Records."""
        completed_rounds = len(records)
        total_retries = 0
        retry_by_cause: dict[str, int] = {}
        for record in records:
            for turn in cast(list[Mapping[str, object]], record["turns"]):
                for attempt in cast(list[Mapping[str, object]], turn["model_attempts"]):
                    cause = attempt["retry_cause"]
                    if isinstance(cause, str):
                        total_retries += 1
                        retry_by_cause[cause] = retry_by_cause.get(cause, 0) + 1
        successful_rounds = sum(
            1
            for record in records
            if cast(Mapping[str, object], record["terminal_outcome"])["kind"]
            == "correct_formal_guess"
        )
        total_score = sum(cast(int, round_values["score"]) for round_values in rounds)
        total_turns = sum(
            cast(int, round_values["turn_count"]) for round_values in rounds
        )
        total_failed_guesses = sum(
            cast(int, round_values["failed_guesses"]) for round_values in rounds
        )
        successful_turns = sum(
            cast(int, round_values["turn_count"])
            for record, round_values in zip(records, rounds, strict=True)
            if cast(Mapping[str, object], record["terminal_outcome"])["kind"]
            == "correct_formal_guess"
        )
        usage_values = [
            cast(Mapping[str, object], round_values["usage"]) for round_values in rounds
        ]
        usage_count_names = (
            "model_attempt_count",
            "provider_call_count",
            "prompt_tokens",
            "output_tokens",
            "reasoning_tokens",
            "answer_tokens",
        )
        usage_totals = {
            name: sum(cast(int, usage[name]) for usage in usage_values)
            for name in usage_count_names
        }
        duration_seconds = round(
            sum(cast(float, usage["duration_seconds"]) for usage in usage_values),
            6,
        )
        output_tokens = usage_totals["output_tokens"]
        compliant_turns = sum(
            1
            for record in records
            for turn in cast(list[Mapping[str, object]], record["turns"])
            if not turn["schema_errors"]
        )
        return {
            "completed_rounds": completed_rounds,
            "successful_rounds": successful_rounds,
            "failed_rounds": completed_rounds - successful_rounds,
            "success_rate": (
                successful_rounds / completed_rounds * 100 if completed_rounds else 0.0
            ),
            "total_score": total_score,
            "average_score": (
                total_score / completed_rounds if completed_rounds else 0.0
            ),
            "total_turns": total_turns,
            "average_turns": (
                total_turns / completed_rounds if completed_rounds else 0.0
            ),
            "average_turns_when_successful": (
                successful_turns / successful_rounds if successful_rounds else 0.0
            ),
            "total_failed_guesses": total_failed_guesses,
            "average_failed_guesses": (
                total_failed_guesses / completed_rounds if completed_rounds else 0.0
            ),
            "schema_compliance_rate": (
                compliant_turns / total_turns if total_turns else None
            ),
            "total_retries": total_retries,
            "retry_by_cause": retry_by_cause,
            "usage": {
                **usage_totals,
                "duration_seconds": duration_seconds,
                "throughput_tokens_per_second": round(
                    output_tokens / duration_seconds if duration_seconds else 0.0,
                    2,
                ),
                "cost": {"usd": None, "pricing_version": None},
            },
        }

    def read_derived_summary(self) -> dict[str, object]:
        """Derive current aggregate statistics from immutable completed Rounds."""
        records = self._completed_rounds()
        rounds = [self.derive_round_values(record) for record in records]
        return self._derived_run_summary(records, rounds)

    def _build_export(self, watermark: int) -> dict[str, object]:
        """Build the versioned portable snapshot from authoritative SQLite data."""
        records = self._completed_rounds()
        rounds = [self.derive_round_values(record) for record in records]
        return {
            "version": BENCHMARK_RUN_EXPORT_VERSION,
            "run": self.read_manifest(),
            "completed_round_records": records,
            "shadow_verdicts": self.read_shadow_verdicts(),
            "derived": {
                "rounds": rounds,
                "summary": self._derived_run_summary(records, rounds),
            },
            "watermark": watermark,
        }

    def _replace_export_atomically(self, document: Mapping[str, object]) -> None:
        """Flush a complete same-directory temporary file before atomic replacement."""
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.run_folder,
                prefix=".results.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(document, temporary, indent=2)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.export_path)
            directory_fd = os.open(self.run_folder, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def ensure_current_export(self) -> Path:
        """Regenerate a missing or stale JSON snapshot without replaying a Round."""
        completed_sequence, export_sequence = self._export_sequences()
        if export_sequence == completed_sequence and self._export_is_current(
            completed_sequence
        ):
            return self.export_path
        document = self._build_export(completed_sequence)
        self._replace_export_atomically(document)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE benchmark_run SET export_sequence = ?",
                (completed_sequence,),
            )
        return self.export_path
