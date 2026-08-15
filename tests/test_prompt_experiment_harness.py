"""Tests for prompt experiment harness (turn-prefix replay)."""

from __future__ import annotations

from eleusis.experiment import (
    ExperimentResults,
    TurnState,
    extract_turn_states,
    run_prompt_experiment,
)


def _make_results_data(num_turns: int = 3) -> ExperimentResults:
    """Build a minimal results structure matching the real format."""
    turns = []
    mainline = "4♠"
    hand = ["2♥", "5♠", "7♦", "10♣"]
    for i in range(num_turns):
        turns.append(
            {
                "turn_number": i + 1,
                "player": "test-model",
                "mainline_state": mainline,
                "hand": hand.copy(),
                "llm_response": {
                    "card": hand[0],
                    "reasoning_summary": f"reasoning {i}",
                    "tentative_rule": "even ranks",
                    "confidence_level": 3,
                    "guess_rule": False,
                },
                "action_result": {
                    "action": "play_card",
                    "card": hand[0],
                    "accepted": True,
                    "success": True,
                },
                "guess_attempt": None,
                "confidence_level_raw": 3,
                "confidence_level": 3,
                "schema_errors": [],
                "tokens": {
                    "output_tokens": 50,
                    "reasoning_tokens": 30,
                    "answer_tokens": 20,
                },
                "retry_count": 0,
                "retry_causes": [],
                "error": None,
            }
        )
        # Extend mainline for next turn
        mainline = f"{mainline} → {hand[0]}"
    return {
        "rounds": [
            {
                "round_number": 1,
                "rule_description": "Only even ranks",
                "rule_code": "return card.rank % 2 == 0",
                "turns": turns,
                "turn_count": num_turns,
                "success": False,
                "score": 10,
                "game_over_reason": "max_turns",
            }
        ],
        "config": {
            "game": {"max_turns": 40, "hand_size": 12, "wrong_guess_penalty": 2},
        },
    }


class TestExtractTurnStates:
    """Turn-prefix states are extracted from results data."""

    def test_extracts_correct_number_of_states(self) -> None:
        """Verify extracts correct number of states."""
        data = _make_results_data(num_turns=3)
        states = extract_turn_states(data, round_index=0)
        assert len(states) == 3

    def test_turn_state_has_required_fields(self) -> None:
        """Verify turn state has required fields."""
        data = _make_results_data(num_turns=1)
        states = extract_turn_states(data, round_index=0)
        state = states[0]
        assert isinstance(state, TurnState)
        assert state.turn_number == 1
        assert state.mainline_state == "4♠"
        assert state.hand == ["2♥", "5♠", "7♦", "10♣"]
        assert state.max_turns == 40
        assert state.hand_size == 12
        assert state.wrong_guess_penalty == 2

    def test_play_history_builds_from_prior_turns(self) -> None:
        """Verify play history builds from prior turns."""
        data = _make_results_data(num_turns=3)
        states = extract_turn_states(data, round_index=0)
        # First turn has no prior history
        assert states[0].play_history == []
        # Second turn has 1 prior play
        assert len(states[1].play_history) == 1
        assert states[1].play_history[0]["card"] == "2♥"
        # Third turn has 2 prior plays
        assert len(states[2].play_history) == 2


class TestRunPromptExperiment:
    """Harness records prompt-variant outputs for comparison."""

    def test_runs_variants_against_turn_states(self) -> None:
        """Verify runs variants against turn states."""
        data = _make_results_data(num_turns=2)
        states = extract_turn_states(data, round_index=0)

        call_log = []

        def variant_a(state: TurnState) -> str:
            return f"variant_a turn {state.turn_number}"

        def variant_b(state: TurnState) -> str:
            return f"variant_b turn {state.turn_number}"

        def fake_generate(prompt: str) -> dict[str, object]:
            call_log.append(prompt)
            return {
                "card": "2♥",
                "reasoning_summary": f"response to: {prompt[:20]}",
                "tentative_rule": "test",
                "confidence_level": 5,
                "guess_rule": False,
            }

        results = run_prompt_experiment(
            turn_states=states,
            variants={"a": variant_a, "b": variant_b},
            generate_fn=fake_generate,
        )

        # Should have results for each turn
        assert len(results) == 2
        # Each turn result has outputs keyed by variant name
        for turn_result in results:
            assert "a" in turn_result["variant_outputs"]
            assert "b" in turn_result["variant_outputs"]
            response = turn_result["variant_outputs"]["a"]["response"]
            assert response is not None
            assert response["card"] == "2♥"

    def test_records_prompts_alongside_responses(self) -> None:
        """Verify records prompts alongside responses."""
        data = _make_results_data(num_turns=1)
        states = extract_turn_states(data, round_index=0)

        def my_variant(state: TurnState) -> str:
            return "my custom prompt"

        def fake_generate(prompt: str) -> dict[str, object]:
            return {
                "card": "5♠",
                "reasoning_summary": "test",
                "tentative_rule": "",
                "confidence_level": 2,
                "guess_rule": False,
            }

        results = run_prompt_experiment(
            turn_states=states,
            variants={"custom": my_variant},
            generate_fn=fake_generate,
        )

        output = results[0]["variant_outputs"]["custom"]
        assert output["prompt"] == "my custom prompt"
        assert output["response"] is not None
        assert output["response"]["card"] == "5♠"

    def test_output_includes_turn_metadata(self) -> None:
        """Verify output includes turn metadata."""
        data = _make_results_data(num_turns=1)
        states = extract_turn_states(data, round_index=0)

        def variant(state: TurnState) -> str:
            return "prompt"

        def fake_generate(prompt: str) -> dict[str, object]:
            return {
                "card": "2♥",
                "reasoning_summary": "",
                "tentative_rule": "",
                "confidence_level": 0,
                "guess_rule": False,
            }

        results = run_prompt_experiment(
            turn_states=states,
            variants={"v": variant},
            generate_fn=fake_generate,
        )

        assert results[0]["turn_number"] == 1
        assert results[0]["mainline_state"] == "4♠"

    def test_handles_generate_errors_gracefully(self) -> None:
        """Verify handles generate errors gracefully."""
        data = _make_results_data(num_turns=1)
        states = extract_turn_states(data, round_index=0)

        def variant(state: TurnState) -> str:
            return "prompt"

        def failing_generate(prompt: str) -> dict[str, object]:
            raise RuntimeError("API error")

        results = run_prompt_experiment(
            turn_states=states,
            variants={"v": variant},
            generate_fn=failing_generate,
        )

        output = results[0]["variant_outputs"]["v"]
        assert output["error"] is not None
        assert "API error" in output["error"]
        assert output["response"] is None
