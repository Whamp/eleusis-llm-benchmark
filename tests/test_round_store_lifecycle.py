"""End-to-end persistence lifecycle tests for authoritative Benchmark Runs."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from eleusis import evaluation_orchestrator, runner
from eleusis.benchmark_run_store import BENCHMARK_RUN_DATABASE_NAME
from eleusis.evaluation_state import _initialize_fresh_state
from eleusis.game.cards import Deck
from eleusis.game.rule_library import RuleLibraryEntry
from tests.conftest import FakeLLMClient, make_action_response
from tests.test_evaluation_orchestrator import _startup


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
        {"round_number": 1, "turn_count": 1, "score": 0}
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
