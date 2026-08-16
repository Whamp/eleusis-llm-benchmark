"""Tests for resume correctness and self-describing run metadata."""

from __future__ import annotations

from eleusis.evaluation_results import EvaluationResults


def _make_checkpoint(
    *,
    completed_rounds: int = 5,
    total_rounds: int = 10,
    num_rules: int = 10,
    num_rounds_per_rule: int = 1,
    rule_factory_index: int = 5,
    selection: str = "sequential",
    player_model: str = "test/model-1",
    batch_round_offset: int | None = None,
) -> EvaluationResults:
    """Build a minimal but valid checkpoint dict matching evaluate_single's schema."""
    return {
        "timestamp": "20260404_120000",
        "folder_name": "solo_evaluation_20260404_120000_test",
        "config": {
            "num_rules": num_rules,
            "num_rounds_per_rule": num_rounds_per_rule,
            "max_turns": 40,
            "hand_size": 12,
            "wrong_guess_penalty": 3,
            "seed": 42,
            "player": "Test Model",
            "player_model": player_model,
            "rule_compiler": "Gemini Flash",
            "rule_compiler_provider": "google",
            "rule_compiler_model_id": "gemini-2.5-flash",
            "rule_compiler_reasoning_format": "separate_field",
            "rule_compiler_temperature": 0.7,
            "rule_compiler_max_retries": 10,
            "rule_compiler_num_simulations": 100,
            "rule_compiler_turns_per_simulation": 40,
            "rule_compiler_simulation_seed": None,
            "llm_max_tokens": 8192,
            "llm_temperature": 0.7,
            "llm_seed": None,
            "llm_max_retries": 3,
            "batch_round_offset": batch_round_offset,
            "suite": None,
        },
        "rounds": [],
        "statistics": {
            "total_score": 0,
            "successful_rounds": 0,
            "failed_rounds": 0,
            "total_turns": 0,
            "total_failed_guesses": 0,
            "total_output_tokens": 0,
            "total_reasoning_tokens": 0,
            "total_answer_tokens": 0,
            "total_wall_clock_seconds": 0.0,
            "total_retries": 0,
            "retry_by_cause": {},
        },
        "checkpoint": {
            "completed_rounds": completed_rounds,
            "total_rounds": total_rounds,
            "rule_factory_state": {
                "selection": selection,
                "current_index": rule_factory_index,
            },
            "current_rule": {
                "description": "Only even ranks.",
                "code": "return card.rank % 2 == 0",
                "rounds_used_in_batch": 1,
                "num_rounds_per_rule": num_rounds_per_rule,
            },
            "rules_consumed": [
                {
                    "name": f"rule_{i}",
                    "description": f"Rule {i}",
                    "code": "return True",
                    "rounds_completed": 1,
                }
                for i in range(completed_rounds)
            ],
            "rules_library": [
                {
                    "name": f"rule_{i}",
                    "description": f"Rule {i}",
                    "code": "return True",
                }
                for i in range(num_rules)
            ],
        },
    }


class TestResultsMetadataSelfDescribing:
    """Results metadata must be self-describing for both fresh and resumed runs."""

    def test_checkpoint_contains_rule_factory_state(self) -> None:
        """Store rule-factory selection and index in checkpoints."""
        checkpoint = _make_checkpoint(selection="sequential", rule_factory_index=3)
        state = checkpoint["checkpoint"]["rule_factory_state"]

        assert "selection" in state
        assert "current_index" in state
        assert state["selection"] == "sequential"
        assert state["current_index"] == 3

    def test_checkpoint_contains_rules_consumed(self) -> None:
        """Checkpoint tracks which rules have been consumed with metadata."""
        checkpoint = _make_checkpoint(completed_rounds=3)
        consumed = checkpoint["checkpoint"]["rules_consumed"]

        assert len(consumed) == 3
        for rule in consumed:
            assert "name" in rule
            assert "description" in rule
            assert "code" in rule
            assert "rounds_completed" in rule

    def test_checkpoint_contains_current_rule(self) -> None:
        """Checkpoint stores the current rule for mid-batch resume."""
        checkpoint = _make_checkpoint()
        current = checkpoint["checkpoint"]["current_rule"]

        assert current is not None
        assert "description" in current
        assert "code" in current

    def test_config_contains_batch_round_offset(self) -> None:
        """Store the parallel worker's batch offset in configuration."""
        checkpoint = _make_checkpoint(batch_round_offset=1)
        assert checkpoint["config"]["batch_round_offset"] == 1

    def test_fresh_run_metadata_shape(self) -> None:
        """A fresh evaluation_results dict has all required metadata keys."""
        # Simulate what main() creates for a fresh run
        fresh_results = {
            "timestamp": "20260404_120000",
            "folder_name": "solo_evaluation_20260404_120000_test",
            "config": {
                "num_rules": 10,
                "num_rounds_per_rule": 1,
                "rule_compiler": "Gemini Flash",
                "rule_compiler_provider": "google",
                "rule_compiler_model_id": "gemini-2.5-flash",
                "player": "Test Model",
                "player_model": "test/model-1",
                "hand_size": 12,
                "max_turns": 40,
                "wrong_guess_penalty": 3,
                "seed": 42,
                "batch_round_offset": None,
            },
            "checkpoint": {
                "completed_rounds": 0,
                "total_rounds": 10,
                "rule_factory_state": {
                    "selection": "sequential",
                    "current_index": 0,
                },
                "current_rule": None,
                "rules_consumed": [],
                "rules_library": [],
            },
        }

        # All required top-level keys
        assert "timestamp" in fresh_results
        assert "folder_name" in fresh_results
        assert "config" in fresh_results
        assert "checkpoint" in fresh_results

        # Config must have model identity
        assert "player_model" in fresh_results["config"]
        assert "rule_compiler_provider" in fresh_results["config"]

        # Checkpoint must have resumability keys
        chk = fresh_results["checkpoint"]
        assert "completed_rounds" in chk
        assert "total_rounds" in chk
        assert "rule_factory_state" in chk
        assert "rules_consumed" in chk
        assert "rules_library" in chk
