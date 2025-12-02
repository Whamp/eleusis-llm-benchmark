"""LLM-powered players for Eleusis: Scientists and Rule-makers."""

import logging
import random
from abc import ABC, abstractmethod

from eleusis.cards import Card
from eleusis.game_engine import Action, NoPlayAction, PlayCardAction, Rule
from eleusis.game_state import GameState
from eleusis.llm_client import HuggingFaceClient
from eleusis.prompts import (
    get_move_selection_prompt,
    get_rule_generation_prompt,
)
from eleusis.rules import LLMGeneratedRule, RuleValidator

logger = logging.getLogger(__name__)


class Player(ABC):
    """Abstract base class for players."""

    def __init__(self, name: str) -> None:
        """Initialize player with name."""
        self.name = name

    @abstractmethod
    def get_action(self, game_state: GameState, can_guess: bool = False) -> Action:
        """Get player's action for their turn."""
        pass


class LLMScientist(Player):
    """Scientist player powered by an LLM."""

    def __init__(
        self,
        name: str,
        llm_client: HuggingFaceClient,
        max_retries: int = 3,
        max_tokens: int = 8192,
    ) -> None:
        """Initialize LLM scientist."""
        super().__init__(name)
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.successful_plays = 0
        self.play_history: list[dict] = []
        self.last_action_response: dict | None = None  # Store LLM response for history

    def get_action(self, game_state: GameState, can_guess: bool = False) -> Action:
        """Get scientist's action using LLM."""
        current_player = game_state.get_current_player()

        # Select a move (play card or no-play)
        # Note: guessing is now handled via guess_rule_if_accepted flag in the action
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
        prompt = get_move_selection_prompt(
            compact_board, hand_dicts, deck_remaining, self.play_history
        )

        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.generate_structured(
                    prompt, max_tokens=self.max_tokens, xml_tag="ACTION"
                )

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
            "reasoning": self.last_action_response.get("reasoning", ""),
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


class LLMRuleMaker:
    """Rule-maker that generates rules using an LLM."""

    def __init__(
        self,
        llm_client: HuggingFaceClient,
        validator: RuleValidator,
        max_attempts: int = 3,
        max_tokens: int = 8192,
    ) -> None:
        """Initialize LLM rule-maker."""
        self.llm_client = llm_client
        self.validator = validator
        self.max_attempts = max_attempts
        self.max_tokens = max_tokens

    def generate_rule(self) -> Rule | None:
        """Generate a valid rule using the LLM.

        Returns PythonRule if code is valid, otherwise falls back to LLMGeneratedRule.
        """
        from eleusis.python_rule import PythonRule

        prompt = get_rule_generation_prompt()

        for attempt in range(self.max_attempts):
            try:
                # Get rule from LLM
                response = self.llm_client.generate(prompt, max_tokens=self.max_tokens)

                # Extract description and code from <RULE> tags
                extracted = self._extract_rule(response)

                if not extracted:
                    logger.warning("Could not extract rule from response")
                    logger.debug(f"Raw response: {response}")
                    continue

                description, code = extracted

                logger.info(f"Generated rule: {description}")
                logger.info(f"Generated Python code:\n{code}")

                # Try to create and validate PythonRule
                python_rule = PythonRule(description, code)
                validation = self.validator.validate_rule(python_rule, num_test_cases=5)

                if validation.valid:
                    logger.info("✓ Python rule is valid, using fast Python evaluation")
                    return python_rule
                else:
                    logger.warning(f"✗ Python rule invalid: {', '.join(validation.issues)}")
                    logger.info("Falling back to LLM evaluation")

                    # Fall back to LLM-based evaluation
                    llm_rule = LLMGeneratedRule(
                        description, self.llm_client, self.llm_client.model_name, code=code
                    )
                    return llm_rule

            except Exception as e:
                logger.error(f"Rule generation attempt {attempt + 1} failed: {e}")

        logger.error("Failed to generate valid rule after all attempts")
        return None

    def _extract_rule(self, response: str) -> tuple[str, str] | None:
        """Extract description and code from <RULE> tags.

        Returns:
            Tuple of (description, code) or None if extraction fails
        """
        import re

        # Look for <RULE>...</RULE> pattern
        rule_match = re.search(r"<RULE>(.*?)</RULE>", response, re.DOTALL | re.IGNORECASE)
        if not rule_match:
            logger.warning("No <RULE> tags found in response")
            return None

        rule_content = rule_match.group(1).strip()

        # Extract <DESCRIPTION>
        desc_match = re.search(
            r"<DESCRIPTION>(.*?)</DESCRIPTION>", rule_content, re.DOTALL | re.IGNORECASE
        )
        # Extract <CODE>
        code_match = re.search(r"<CODE>(.*?)</CODE>", rule_content, re.DOTALL | re.IGNORECASE)

        if not desc_match or not code_match:
            logger.warning("Missing <DESCRIPTION> or <CODE> tags in rule")
            return None

        description = desc_match.group(1).strip()
        code = code_match.group(1).strip()

        return description, code


class RandomScientist(Player):
    """Scientist that plays randomly (for testing)."""

    def __init__(self, name: str) -> None:
        """Initialize random scientist."""
        super().__init__(name)

    def get_action(self, game_state: GameState, can_guess: bool = False) -> Action:
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
