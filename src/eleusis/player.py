"""LLM-based player for Eleusis card game."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from typing_extensions import TypedDict

from eleusis.game.cards import Card, Suit
from eleusis.llm.base import TruncationError

__all__ = ["LLMScientist"]

if TYPE_CHECKING:
    from eleusis.evaluation_results import ModelAttemptRecord, ProviderCallRecord
    from eleusis.game.engine import Action, GameEngine
    from eleusis.game.state import GameState
    from eleusis.llm.base import BaseLLMClient, LLMCallMetrics

logger = logging.getLogger(__name__)

_TRUNCATION_RETRY_HINT = (
    "\n\nYour last response hit the output token limit. "
    "Output ONLY the <ACTION> XML block with no reasoning."
)
_CARD_PARSE_RETRY_HINT = (
    "\n\nThe card value could not be parsed. Use exact symbol format: 5♥, K♠, A♦, etc."
)
_GENERIC_RETRY_HINT = "\n\nIMPORTANT: DO NOT REASON TOO LONG ABOUT THIS."


def _random_state_to_json(value: object) -> object:
    """Convert nested RNG-state tuples into JSON-compatible arrays."""
    if isinstance(value, tuple):
        return [_random_state_to_json(item) for item in value]
    return value


def _random_state_from_json(value: object) -> object:
    """Convert nested JSON arrays back into tuples accepted by random.Random."""
    if isinstance(value, list):
        return tuple(_random_state_from_json(item) for item in value)
    return value


class PlayHistoryEntry(TypedDict):
    """Card outcome and reasoning summary retained for future prompts."""

    card: str
    accepted: bool
    reasoning_summary: str


class RetryCause(TypedDict):
    """Cause associated with one failed model action attempt."""

    attempt: int
    cause: str


class LLMScientist:
    """Scientist player using LLM for decision making."""

    def __init__(
        self,
        name: str,
        llm_client: BaseLLMClient,
        max_retries: int = 3,
        engine: GameEngine | None = None,
        max_turns: int = 40,
        rng: random.Random | None = None,
    ) -> None:
        """Initialize scientist with name and LLM client."""
        self.name = name
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.engine = engine
        self.max_turns = max_turns
        self.rng = rng or random.Random()
        self.play_history: list[PlayHistoryEntry] = []
        self.last_action_response: dict[str, object] | None = None
        self.last_prompt: str | None = None
        # Retry tracking (reset each turn)
        self.last_retry_count: int = 0
        self.last_retry_causes: list[RetryCause] = []
        self.last_model_attempts: list[ModelAttemptRecord] = []

    def snapshot_scientist_continuation(self) -> dict[str, object]:
        """Capture prompt history and deterministic fallback RNG continuation."""
        return {
            "name": self.name,
            "max_retries": self.max_retries,
            "max_turns": self.max_turns,
            "play_history": [dict(entry) for entry in self.play_history],
            "rng_state": _random_state_to_json(self.rng.getstate()),
        }

    @classmethod
    def restore_scientist_continuation(
        cls,
        payload: Mapping[str, object],
        *,
        llm_client: BaseLLMClient,
        engine: GameEngine,
    ) -> LLMScientist:
        """Build a fresh scientist with restored prompt history and RNG state."""
        rng = random.Random()
        rng.setstate(
            cast(
                tuple[int, tuple[int, ...], float | None],
                _random_state_from_json(payload["rng_state"]),
            )
        )
        scientist = cls(
            cast(str, payload["name"]),
            llm_client,
            max_retries=cast(int, payload["max_retries"]),
            engine=engine,
            max_turns=cast(int, payload["max_turns"]),
            rng=rng,
        )
        scientist.play_history = [
            cast(PlayHistoryEntry, dict(entry))
            for entry in cast(list[Mapping[str, object]], payload["play_history"])
        ]
        return scientist

    def get_action(self, game_state: GameState) -> Action:
        """Get an action for the current game state."""
        return self._select_move(game_state)

    @staticmethod
    def _provider_call_record(
        metric: LLMCallMetrics,
        call_number: int,
    ) -> ProviderCallRecord:
        """Project one observable client metric into Provider Call evidence."""
        return {
            "call_number": call_number,
            "provider": metric.provider,
            "model": metric.model_name,
            "timestamp": metric.timestamp,
            "duration_seconds": metric.duration_seconds,
            "finish_reason": metric.finish_reason,
            "is_continuation": metric.is_continuation,
            "continuation_depth": metric.continuation_depth,
            "token_metrics": {
                "prompt_tokens": metric.prompt_tokens,
                "output_tokens": metric.output_tokens,
                "reasoning_tokens": metric.reasoning_tokens,
                "answer_tokens": metric.answer_tokens,
            },
        }

    def _record_model_attempt(
        self,
        *,
        attempt_number: int,
        prompt: str,
        interpretation: str,
        retry_cause: str | None,
        started_at: float,
        call_metrics_before: int,
    ) -> None:
        """Retain one complete prompt submission as Model Attempt evidence."""
        provider_calls = [
            self._provider_call_record(metric, call_number)
            for call_number, metric in enumerate(
                self.llm_client.call_metrics[call_metrics_before:],
                start=1,
            )
        ]
        token_metrics = {
            name: sum(call["token_metrics"][name] for call in provider_calls)
            for name in (
                "prompt_tokens",
                "output_tokens",
                "reasoning_tokens",
                "answer_tokens",
            )
        }
        reasoning_trace = "\n\n".join(
            metric.reasoning_text
            for metric in self.llm_client.call_metrics[call_metrics_before:]
            if metric.reasoning_text
        )
        self.last_model_attempts.append(
            cast(
                "ModelAttemptRecord",
                {
                    "attempt_number": attempt_number,
                    "prompt": prompt,
                    "raw_completion": self.llm_client.last_raw_completion,
                    "structured_completion": (
                        dict(self.last_action_response)
                        if self.last_action_response is not None
                        else None
                    ),
                    "interpretation": interpretation,
                    "retry_cause": retry_cause,
                    "started_at": started_at,
                    "duration_seconds": max(0.0, time.time() - started_at),
                    "provider": self.llm_client.provider_name,
                    "model": self.llm_client.model_name,
                    "finish_reason": (
                        provider_calls[-1]["finish_reason"] if provider_calls else None
                    ),
                    "token_metrics": token_metrics,
                    "reasoning_text": reasoning_trace or None,
                    "provider_calls": provider_calls,
                },
            )
        )

    def _select_move(self, game_state: GameState) -> Action:
        """Select a card to play using LLM retries and a random fallback."""
        from eleusis.game.engine import PlayCardAction

        self.last_action_response = None
        self.last_retry_count = 0
        self.last_retry_causes = []
        self.last_model_attempts = []
        if self.engine is None:
            raise RuntimeError(
                "LLMScientist requires a GameEngine before selecting a move"
            )

        hand_cards = game_state.player.hand.get_all_cards()
        if not hand_cards:
            raise RuntimeError("LLMScientist cannot select a move from an empty hand")
        base_prompt = self._build_action_prompt(game_state, hand_cards)

        last_cause: str | None = None
        for attempt in range(self.max_retries):
            attempt_number = attempt + 1
            prompt = self._retry_prompt(base_prompt, attempt, last_cause)
            self.last_prompt = prompt
            self.last_action_response = None
            self.llm_client.last_raw_completion = None
            started_at = time.time()
            call_metrics_before = len(self.llm_client.call_metrics)
            try:
                card, card_value = self._request_action_card(prompt, hand_cards)
                if card:
                    self._record_model_attempt(
                        attempt_number=attempt_number,
                        prompt=prompt,
                        interpretation="usable_action",
                        retry_cause=None,
                        started_at=started_at,
                        call_metrics_before=call_metrics_before,
                    )
                    return PlayCardAction(card)
                cause = "card_parse_error"
                interpretation = "card_parse_error"
                logger.warning(
                    f"{self.name} attempt {attempt_number}: {cause} - "
                    f"card='{card_value}'"
                )
            except TruncationError as error:
                cause = "max_token_reached"
                interpretation = "truncated"
                logger.warning(
                    f"{self.name} attempt {attempt_number}: {cause} - {error}"
                )
            except (json.JSONDecodeError, TypeError) as error:
                cause = "structured_response_parse_error"
                interpretation = "structured_response_parse_error"
                logger.warning(
                    f"{self.name} attempt {attempt_number}: {cause} - "
                    f"{type(error).__name__}: {error}"
                )
            # The player retry boundary records arbitrary provider failures.
            except Exception as error:  # ruff: ignore[blind-except]
                cause = "other_error"
                interpretation = "provider_error"
                logger.warning(
                    f"{self.name} attempt {attempt_number}: {cause} -"
                    f" {type(error).__name__}: {error}"
                )
            self._record_model_attempt(
                attempt_number=attempt_number,
                prompt=prompt,
                interpretation=interpretation,
                retry_cause=cause,
                started_at=started_at,
                call_metrics_before=call_metrics_before,
            )
            last_cause = cause
            self.last_retry_count = attempt_number
            self.last_retry_causes.append({"attempt": attempt_number, "cause": cause})

        logger.warning(
            f"{self.name} using random fallback after {self.max_retries} failed"
            " attempts"
        )
        self.last_action_response = None
        return PlayCardAction(self.rng.choice(hand_cards))

    def _build_action_prompt(
        self,
        game_state: GameState,
        hand_cards: list[Card],
    ) -> str:
        """Build the model prompt for the current game and scoring state."""
        from eleusis.prompts.action import ActionPromptContext, get_action_prompt

        if self.engine is None:
            raise RuntimeError("Action prompt requires an attached GameEngine")
        return get_action_prompt(
            ActionPromptContext(
                compact_board=game_state.to_compact_string(),
                hand_cards=[card.to_dict() for card in hand_cards],
                play_history=self.play_history,
                failed_guesses=game_state.failed_rule_guesses,
                current_turn=game_state.turn_number,
                max_turns=self.max_turns,
                failed_guess_count=self.engine.failed_guess_count,
                hand_size=self.engine.hand_size,
                wrong_guess_penalty=self.engine.wrong_guess_penalty,
            )
        )

    @staticmethod
    def _retry_prompt(base_prompt: str, attempt: int, last_cause: str | None) -> str:
        """Append the retry guidance appropriate to the preceding failure."""
        if attempt == 0:
            return base_prompt
        if last_cause == "max_token_reached":
            return base_prompt + _TRUNCATION_RETRY_HINT
        if last_cause == "card_parse_error":
            return base_prompt + _CARD_PARSE_RETRY_HINT
        return base_prompt + _GENERIC_RETRY_HINT

    def _request_action_card(
        self,
        prompt: str,
        hand_cards: list[Card],
    ) -> tuple[Card | None, str]:
        """Request one structured action and parse its card from the hand."""
        response = self.llm_client.generate(prompt, xml_tag="ACTION", return_dict=True)
        self.last_action_response = response
        card_value_raw = response.get("card", "")
        card_value = card_value_raw.strip() if isinstance(card_value_raw, str) else ""
        card = self._parse_card(card_value, hand_cards)
        if card:
            logger.info(f"{self.name} plays {card}")
            tentative = response.get("tentative_rule", "")
            if isinstance(tentative, str) and tentative:
                logger.debug(f"{self.name}'s tentative rule: {tentative}")
        return card, card_value

    def record_action_result(self, result: Mapping[str, object]) -> None:
        """Record the result of an action in play history."""
        if result.get("success") and "card" in result:
            reasoning_value = (
                self.last_action_response.get("reasoning_summary", "")
                if self.last_action_response
                else ""
            )
            card_value = result["card"]
            accepted_value = result.get("accepted", False)
            if isinstance(card_value, str) and isinstance(accepted_value, bool):
                self.record_play(
                    card_str=card_value,
                    accepted=accepted_value,
                    reasoning_summary=(
                        reasoning_value if isinstance(reasoning_value, str) else ""
                    ),
                )

    def _parse_card(self, card_value: str, hand_cards: list[Card]) -> Card | None:
        """Parse card value string to Card object from hand."""
        suit_map = {
            "♥": Suit.HEARTS,
            "♦": Suit.DIAMONDS,
            "♣": Suit.CLUBS,
            "♠": Suit.SPADES,
            "hearts": Suit.HEARTS,
            "diamonds": Suit.DIAMONDS,
            "clubs": Suit.CLUBS,
            "spades": Suit.SPADES,
        }
        rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}

        if not card_value:
            return None

        card_value = card_value.strip()

        suit = None
        rank_str = card_value

        for symbol, s in suit_map.items():
            if symbol in card_value:
                suit = s
                rank_str = card_value.replace(symbol, "").strip()
                break

        if not suit:
            return None

        rank_str = rank_str.upper()
        if rank_str in rank_map:
            rank = rank_map[rank_str]
        else:
            try:
                rank = int(rank_str)
            except ValueError:
                return None

        for card in hand_cards:
            if card.rank == rank and card.suit == suit:
                return card

        return None

    def record_play(
        self, card_str: str, accepted: bool, reasoning_summary: str = ""
    ) -> None:
        """Record a play attempt in history."""
        self.play_history.append(
            {
                "card": card_str,
                "accepted": accepted,
                "reasoning_summary": reasoning_summary,
            }
        )
