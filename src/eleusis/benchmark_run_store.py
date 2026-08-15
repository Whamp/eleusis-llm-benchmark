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
)
from eleusis.evaluation_results import TurnRecord
from eleusis.round_continuation import (
    capture_round_continuation,
    validate_round_continuation_document,
)
from eleusis.round_record import (
    append_round_record_turn,
    complete_round_record,
    create_active_round_record,
    validate_round_record_document,
)

if TYPE_CHECKING:
    from eleusis.round_execution import RoundRuntime

BENCHMARK_RUN_DATABASE_NAME = "benchmark_run.sqlite3"


class BenchmarkRunStoreError(RuntimeError):
    """Raised when the authoritative Benchmark Run store rejects an operation."""


@dataclass(frozen=True)
class ActiveStoredRound:
    """Validated active Round Record and its hidden continuation checkpoint."""

    record: dict[str, object]
    continuation: dict[str, object]


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
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT database_version FROM benchmark_run"
            ).fetchone()
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
        if not isinstance(document, dict):
            raise BenchmarkRunStoreError(
                "Benchmark Run store manifest invalid: document must be an object"
            )
        return cast(dict[str, object], document)

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

    def read_active_round(self, round_number: int) -> ActiveStoredRound | None:
        """Read and validate the active Round Record and continuation checkpoint."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT record_document, continuation_document
                FROM rounds
                WHERE round_number = ? AND status = 'active'
                """,
                (round_number,),
            ).fetchone()
        if row is None:
            return None
        return ActiveStoredRound(
            record=validate_round_record_document(
                self._decode_document(cast(str, row["record_document"]))
            ),
            continuation=validate_round_continuation_document(
                self._decode_document(cast(str, row["continuation_document"]))
            ),
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

    def _export_is_current(self, completed_sequence: int) -> bool:
        """Check whether the portable export has the authoritative watermark."""
        if not self.export_path.is_file():
            return False
        try:
            document = json.loads(self.export_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return (
            isinstance(document, dict)
            and document.get("version") == BENCHMARK_RUN_EXPORT_VERSION
            and document.get("watermark") == completed_sequence
        )

    @staticmethod
    def _derived_round_values(record: Mapping[str, object]) -> dict[str, int]:
        """Recompute simple tracer metrics from authoritative terminal facts."""
        turns = cast(list[Mapping[str, object]], record["turns"])
        outcome = cast(Mapping[str, object], record["terminal_outcome"])
        settings = cast(Mapping[str, object], record["settings"])
        last_turn = turns[-1]
        post_state = cast(Mapping[str, object], last_turn["post_card_state"])
        failed_guesses = cast(list[object], post_state["failed_rule_guesses"])
        penalty = cast(int, settings["wrong_guess_penalty"]) * len(failed_guesses)
        score = -penalty
        if outcome["kind"] == "correct_formal_guess":
            score = cast(int, settings["max_turns"]) - len(turns) + 1 - penalty
        return {
            "round_number": cast(int, record["scheduled_round_number"]),
            "turn_count": len(turns),
            "score": score,
        }

    def _build_export(self, watermark: int) -> dict[str, object]:
        """Build the versioned portable snapshot from authoritative SQLite data."""
        records = self._completed_rounds()
        return {
            "version": BENCHMARK_RUN_EXPORT_VERSION,
            "run": self.read_manifest(),
            "completed_round_records": records,
            "derived": {
                "rounds": [self._derived_round_values(record) for record in records]
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
