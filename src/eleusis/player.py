"""LLM-powered players for Eleusis."""

import logging
import random
from abc import ABC, abstractmethod

from eleusis.cards import Card
from eleusis.game_engine import Action, NoPlayAction, PlayCardAction
from eleusis.game_state import GameState
from eleusis.llm_client import BaseLLMClient
from eleusis.prompts import get_action_selection_prompt

logger = logging.getLogger(__name__)


class Player(ABC):
    """Abstract base class for players."""

    def __init__(self, name: str) -> None:
        """Initialize player with name."""
        self.name = name

    @abstractmethod
    def get_action(self, game_state: GameState) -> Action:
        """Get player's action for their turn."""
        pass


class LLMScientist(Player):
    """Scientist player powered by an LLM."""

    def __init__(
        self,
        name: str,
        llm_client: BaseLLMClient,
        max_retries: int = 3,
    ) -> None:
        """Initialize LLM scientist."""
        super().__init__(name)
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.successful_plays = 0
        self.play_history: list[dict] = []
        self.last_action_response: dict | None = None  # Store LLM response for history

    def get_action(self, game_state: GameState) -> Action:
        """Get scientist's action using LLM."""
        current_player = game_state.get_current_player()
        return self._select_move(game_state, current_player)

    def should_guess_now(self) -> bool:
        """Check if we should guess based on the last action's guess_rule_if_accepted flag."""
        if not self.last_action_response:
            return False
        return self.last_action_response.get("guess_rule_if_accepted", False)

    def get_guess_from_last_action(self) -> str | None:
        """Get the tentative_rule from the last action to use as a guess."""
        if not self.last_action_response:
            return None
        return self.last_action_response.get("tentative_rule", "")

    def _select_move(self, game_state: GameState, current_player) -> Action:
        """Select a card to play or declare no-play."""
        hand_cards = current_player.hand.get_all_cards()
        if not hand_cards:
            # Must guess if hand is empty (handled by game engine)
            return NoPlayAction()

        hand_dicts = [c.to_dict() for c in hand_cards]
        compact_board = game_state.to_compact_string()
        deck_remaining = game_state.deck.remaining_count()
        failed_guesses = game_state.failed_rule_guesses
        prompt = get_action_selection_prompt(
            compact_board, hand_dicts, deck_remaining, self.play_history, failed_guesses
        )

        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.generate(prompt, xml_tag="ACTION", return_dict=True)

                # Store response for history tracking
                self.last_action_response = response

                # Parse action field (now contains card or "no_play")
                action_value = response.get("action", "").strip()

                if action_value.lower() == "no_play":
                    logger.info(f"{self.name} declares no-play")
                    tentative = response.get("tentative_rule", "")
                    if tentative:
                        logger.debug(f"{self.name}'s tentative rule: {tentative}")
                    return NoPlayAction()

                else:
                    # action_value should be a card string like "5♥"
                    card = self._parse_card(action_value, hand_cards)
                    if card:
                        logger.info(f"{self.name} plays {card}")
                        tentative = response.get("tentative_rule", "")
                        if tentative:
                            logger.debug(f"{self.name}'s tentative rule: {tentative}")
                        return PlayCardAction(card)

            except Exception as e:
                logger.warning(f"Move selection attempt {attempt + 1} failed: {e}")

        # Fallback: play random card
        logger.warning(f"{self.name} using random fallback")
        return PlayCardAction(random.choice(hand_cards))

    def _parse_card(self, card_str: str, hand_cards: list[Card]) -> Card | None:
        """Parse card string like '5♥' and find in hand."""
        for card in hand_cards:
            if str(card) == card_str:
                return card
        return None

    def record_action_result(self, result: dict) -> None:
        """Record the action and its result for learning from history.

        Args:
            result: The result dictionary from game engine's play_turn()
        """
        if not self.last_action_response:
            return

        # Build history entry from LLM response and game result
        history_entry = {
            "action": self.last_action_response.get("action", "unknown"),
            "reasoning_summary": self.last_action_response.get("reasoning_summary", ""),
        }

        # Add result-specific fields
        if "card" in result:
            history_entry["card"] = result["card"]
            history_entry["accepted"] = result.get("accepted", False)
            if result.get("accepted"):
                self.successful_plays += 1
        elif "correct" in result:
            history_entry["correct"] = result["correct"]

        self.play_history.append(history_entry)
        self.last_action_response = None  # Clear for next turn


class RandomScientist(Player):
    """Scientist that plays randomly (for testing)."""

    def __init__(self, name: str) -> None:
        """Initialize random scientist."""
        super().__init__(name)

    def get_action(self, game_state: GameState) -> Action:
        """Get random action."""
        current_player = game_state.get_current_player()
        hand_cards = current_player.hand.get_all_cards()

        if not hand_cards:
            return NoPlayAction()

        # 80% play random card, 20% no-play
        if random.random() < 0.8:
            return PlayCardAction(random.choice(hand_cards))
        else:
            return NoPlayAction()
