"""End-to-end persistence lifecycle tests for authoritative Benchmark Runs."""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest
from pytest import MonkeyPatch

from eleusis import benchmark_run_store, evaluation_orchestrator, runner
from eleusis.benchmark_run_store import (
    BENCHMARK_RUN_DATABASE_NAME,
    BenchmarkRunStore,
    BenchmarkRunStoreError,
)
from eleusis.evaluation_state import _initialize_fresh_state
from eleusis.game.cards import Deck
from eleusis.game.rule_library import RuleLibraryEntry
from eleusis.round_continuation import RoundContinuationIncompatibilityError
from eleusis.round_execution import execute_round_turn, execute_round_turns
from tests.conftest import FakeLLMClient, make_action_response
from tests.test_evaluation_orchestrator import _startup
from tests.test_round_continuation import _build_round_runtime
from tests.test_round_record import _manifest


def test_one_turn_round_uses_complete_authoritative_store_lifecycle(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist setup, one Turn, terminal record, and a regenerable export."""
    monkeypatch.chdir(tmp_path)
    startup = _startup()
    startup.config["game"]["max_turns"] = 1
    startup.game_config["max_turns"] = 1
    startup.config["game"]["seed"] = None
    startup.game_config["seed"] = None
    rule_entry: RuleLibraryEntry = {
        "name": "only_red",
        "description": "Only red cards.",
        "code": "return card.color == 'red'",
    }
    Path("rules.json").write_text(json.dumps({"rules": [rule_entry]}))

    state = _initialize_fresh_state(startup)
    run_store = state.run_store
    assert run_store is not None
    assert isinstance(startup.game_config["seed"], int)
    effective_seed = startup.game_config["seed"]
    rule_hash = int(hashlib.md5(rule_entry["code"].encode()).hexdigest(), 16)
    round_seed = (effective_seed + (rule_hash & 0xFFFFFFFF)) & 0xFFFFFFFF
    deck = Deck()
    deck.shuffle(seed=round_seed)
    selected_card = deck.draw()

    scientist_client = FakeLLMClient([make_action_response(str(selected_card))])
    compiler_client = FakeLLMClient()
    original_generate = scientist_client.generate

    def generate_after_initial_checkpoint(
        prompt: str,
        xml_tag: str | None = None,
        return_dict: bool = False,
    ) -> str | dict[str, object]:
        active = run_store.read_active_round(1)
        assert active is not None
        assert active.record["turns"] == []
        assert active.continuation["next_turn_index"] == 0
        return original_generate(prompt, xml_tag, return_dict)

    monkeypatch.setattr(scientist_client, "generate", generate_after_initial_checkpoint)
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (compiler_client, scientist_client),
    )

    evaluation_orchestrator._run_evaluation_round(state, 1)

    run_folder = tmp_path / "results" / state.folder_name
    database_path = run_folder / BENCHMARK_RUN_DATABASE_NAME
    assert database_path.is_file()
    assert list(run_folder.glob("*.sqlite3")) == [database_path]

    manifest = run_store.read_manifest()
    assert manifest["run_id"]
    assert manifest["versions"] == {
        "database": 1,
        "manifest": 1,
        "round_record": 1,
        "round_checkpoint": 1,
        "export": 1,
    }
    effective_settings = manifest["effective_settings"]
    assert isinstance(effective_settings, dict)
    assert effective_settings["game_seed"] == effective_seed
    assert manifest["schedule"] == [
        {
            "round_number": 1,
            "rule_name": "only_red",
            "rule_description": "Only red cards.",
            "rule_code": "return card.color == 'red'",
            "batch_round_index": 0,
        }
    ]
    model_identity = manifest["model_identity"]
    compiler_identity = manifest["compiler_identity"]
    source_provenance = manifest["source_provenance"]
    assert isinstance(model_identity, dict)
    assert isinstance(compiler_identity, dict)
    assert isinstance(source_provenance, dict)
    assert model_identity["model_key"] == "test-model"
    assert compiler_identity["model_id"] == "compiler"
    assert manifest["prompt_identities"]
    assert source_provenance["revision"]

    completed = run_store.read_completed_round(1)
    assert completed["run_id"] == manifest["run_id"]
    assert completed["scheduled_round_number"] == 1
    assert completed["terminal_outcome"] == {"kind": "turn_limit"}
    completed_settings = cast(dict[str, object], completed["settings"])
    scientific_config = cast(dict[str, object], manifest["scientific_config"])
    assert completed_settings["llm"] == scientific_config["llm"]
    assert completed_settings["rule_compiler"] == scientific_config["rule_compiler"]
    turns = completed["turns"]
    assert isinstance(turns, list)
    assert len(turns) == 1
    turn = turns[0]
    assert isinstance(turn, dict)
    assert list(turn) == [
        "turn_number",
        "pre_decision_state",
        "model_attempts",
        "final_decision",
        "card_outcome",
        "post_card_state",
        "guess_attempt",
        "schema_errors",
    ]
    attempts = turn["model_attempts"]
    final_decision = turn["final_decision"]
    pre_decision_state = turn["pre_decision_state"]
    card_outcome = turn["card_outcome"]
    assert isinstance(attempts, list)
    assert isinstance(final_decision, dict)
    assert isinstance(pre_decision_state, dict)
    assert isinstance(card_outcome, dict)
    assert len(attempts) == 1
    attempt = attempts[0]
    assert isinstance(attempt, dict)
    assert attempt["prompt"] == scientist_client.prompts_seen[0]
    assert attempt["interpretation"] == "usable_action"
    assert final_decision["origin"] == "model_attempt"
    assert final_decision["selected_card"] in pre_decision_state["hand"]
    assert card_outcome["accepted"] in (True, False)
    assert card_outcome["replacement_draw"] is not None
    assert run_store.read_active_round(1) is None

    export_path = run_folder / "results.json"
    exported = json.loads(export_path.read_text())
    assert exported["version"] == 1
    assert exported["watermark"] > 0
    assert exported["completed_round_records"] == [completed]
    assert exported["derived"]["rounds"] == [
        {
            "round_number": 1,
            "turn_count": 1,
            "score": 0,
            "floored_score": 0,
            "no_stakes_score": 0,
            "first_correct_turn": None,
            "failed_guesses": 0,
            "schema_compliance_rate": 1.0,
            "usage": {
                "model_attempt_count": 1,
                "provider_call_count": 1,
                "prompt_tokens": 100,
                "output_tokens": 50,
                "reasoning_tokens": 30,
                "answer_tokens": 20,
                "duration_seconds": 0.1,
                "throughput_tokens_per_second": 500.0,
                "cost": {"usd": None, "pricing_version": None},
            },
        }
    ]
    assert "continuation" not in exported
    assert "continuation" not in completed

    export_path.unlink()
    regenerated_path = run_store.ensure_current_export()
    assert regenerated_path == export_path
    assert json.loads(export_path.read_text())["completed_round_records"] == [completed]

    stale = json.loads(export_path.read_text())
    stale["watermark"] = 0
    export_path.write_text(json.dumps(stale))
    run_store.ensure_current_export()
    assert json.loads(export_path.read_text())["watermark"] == exported["watermark"]

    old_complete_export = json.loads(export_path.read_text())
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE benchmark_run SET export_sequence = -1")
    with monkeypatch.context() as replacement_failure:
        replacement_failure.setattr(
            benchmark_run_store.os,
            "replace",
            lambda _source, _destination: (_ for _ in ()).throw(
                OSError("replacement interrupted")
            ),
        )
        with pytest.raises(OSError, match="replacement interrupted"):
            run_store.ensure_current_export()
    assert json.loads(export_path.read_text()) == old_complete_export
    run_store.ensure_current_export()
    assert run_store.read_export_status().is_current is True

    with sqlite3.connect(database_path) as connection:
        with pytest.raises(
            sqlite3.IntegrityError, match="immutable completed Round Record"
        ):
            connection.execute(
                "UPDATE rounds SET record_document = '{}' WHERE round_number = 1"
            )
        with pytest.raises(
            sqlite3.IntegrityError, match="immutable completed Round Record"
        ):
            connection.execute("DELETE FROM rounds WHERE round_number = 1")


def test_unexpected_action_error_leaves_only_initial_checkpoint(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unexpected player errors propagate without committing a Turn or outcome."""
    runtime = _build_round_runtime()
    store = BenchmarkRunStore.create(tmp_path / "run", _manifest(runtime))
    store.start_round(runtime, effective_round_seed=8675309, batch_round_index=0)
    initial = store.read_active_round(runtime.round_number)
    assert initial is not None
    runtime.completed_turn_committer = store.commit_completed_turn

    def fail_get_action(_state: object) -> object:
        raise RuntimeError("unexpected scientist failure")

    monkeypatch.setattr(runtime.scientist, "get_action", fail_get_action)
    with pytest.raises(RuntimeError, match="unexpected scientist failure"):
        execute_round_turns(runtime)

    active = store.read_active_round(runtime.round_number)
    assert active == initial
    assert active.record["terminal_outcome"] is None
    assert store.read_completed_rounds() == []


def test_active_round_rejects_cross_document_round_identity(tmp_path: Path) -> None:
    """Resume rejects SQL, Round Record, and continuation identity disagreement."""
    runtime = _build_round_runtime()
    store = BenchmarkRunStore.create(tmp_path / "run", _manifest(runtime))
    store.start_round(runtime, effective_round_seed=8675309, batch_round_index=0)
    with sqlite3.connect(store.database_path) as connection:
        row = connection.execute(
            "SELECT continuation_document FROM rounds WHERE round_number = ?",
            (runtime.round_number,),
        ).fetchone()
        assert row is not None
        continuation = json.loads(row[0])
        continuation["round"]["number"] = 99
        encoded = json.dumps(continuation, separators=(",", ":"), sort_keys=True)
        connection.execute(
            "UPDATE rounds SET continuation_document = ? WHERE round_number = ?",
            (encoded, runtime.round_number),
        )
    row_before = store.database_path.read_bytes()

    with pytest.raises(BenchmarkRunStoreError, match="Round identity"):
        store.read_resumable_round()

    assert runtime.scientist_client.call_metrics == []
    assert store.database_path.read_bytes() == row_before


def test_initial_checkpoint_rejects_third_physical_card_copy(tmp_path: Path) -> None:
    """Malformed live setup cannot commit an initial active Round row."""
    runtime = _build_round_runtime()
    duplicate = runtime.game_state.player.hand.get_all_cards()[0]
    runtime.game_state.deck = Deck.restore_deck_cards(
        [duplicate.to_canonical_card_data(), duplicate.to_canonical_card_data()]
    )
    store = BenchmarkRunStore.create(tmp_path / "run", _manifest(runtime))

    with pytest.raises(RoundContinuationIncompatibilityError, match="at most twice"):
        store.start_round(runtime, effective_round_seed=8675309, batch_round_index=0)

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM rounds").fetchone() == (0,)


def test_export_derives_round_and_aggregate_schema_compliance(
    tmp_path: Path,
) -> None:
    """Compliant and noncompliant Turn evidence drives both export rates."""
    runtime = _build_round_runtime()
    store = BenchmarkRunStore.create(tmp_path / "run", _manifest(runtime))
    store.start_round(runtime, effective_round_seed=8675309, batch_round_index=0)
    client = runtime.scientist_client
    assert isinstance(client, FakeLLMClient)
    turns = []
    for turn_index, confidence in enumerate((3, "invalid")):
        selected = runtime.game_state.player.hand.get_all_cards()[0]
        client.responses.append(
            make_action_response(str(selected), confidence_level=confidence)
        )
        turn, _result = execute_round_turn(runtime, turn_index)
        turns.append(turn)
        store.commit_completed_turn(runtime, turns)
    store.complete_round(runtime.round_number, game_over_reason="abandoned")

    completed = store.read_completed_round(runtime.round_number)
    completed_turns = cast(list[dict[str, object]], completed["turns"])
    assert [turn["schema_errors"] for turn in completed_turns] == [
        [],
        ["confidence_type"],
    ]
    exported = json.loads(store.export_path.read_text())
    assert exported["derived"]["rounds"][0]["schema_compliance_rate"] == pytest.approx(
        0.5
    )
    assert exported["derived"]["summary"]["schema_compliance_rate"] == pytest.approx(
        0.5
    )


def test_malformed_database_reports_store_schema_error(tmp_path: Path) -> None:
    """Resume translates SQLite schema defects into a path-specific store error."""
    run_folder = tmp_path / "malformed-run"
    run_folder.mkdir()
    database_path = run_folder / BENCHMARK_RUN_DATABASE_NAME
    sqlite3.connect(database_path).close()

    with pytest.raises(
        BenchmarkRunStoreError,
        match=rf"{database_path}.*schema defect.*benchmark_run",
    ):
        BenchmarkRunStore(run_folder)
