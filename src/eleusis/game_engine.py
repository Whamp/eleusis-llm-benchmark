"""Game engine for Eleusis: turn logic, game flow, and scoring."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from eleusis.cards import Card
from eleusis.game_state import GameState, PlayerState

logger = logging.getLogger(__name__)


# Action types that players can take
@dataclass
class PlayCardAction:
    """Action to play a card from hand."""

    card: Card


@dataclass
class NoPlayAction:
    """Action to declare no playable cards."""

    pass


@dataclass
class GuessRuleAction:
    """Action to guess the secret rule."""

    guess_text: str


Action = PlayCardAction | NoPlayAction | GuessRuleAction


class Rule(ABC):
    """Abstract base class for game rules."""

    @abstractmethod
    def evaluate(self, card: Card, mainline: list[Card]) -> bool:
        """Evaluate if a card is 'in' (accepted) given the current mainline."""
        pass

    @abstractmethod
    def description(self) -> str:
        """Return human-readable description of the rule."""
        pass


class GameEngine:
    """Orchestrates Eleusis game progression."""

    def __init__(
        self,
        game_state: GameState,
        rule: Rule,
        rule_validator=None,
        cards_per_scientist: int = 12,
        correct_guess_bonus: int = -3,
        card_reject_penalty: int = 2,
        no_play_incorrect_penalty: int = 4,
        no_play_correct_reduction: int = 4,
    ) -> None:
        """Initialize game engine with state and rule."""
        self.state = game_state
        self.rule = rule
        self.rule_validator = rule_validator
        self.rule_guessed = False
        self.winning_guesser: str | None = None
        self.cards_per_scientist = cards_per_scientist
        self.correct_guess_bonus = correct_guess_bonus
        self.card_reject_penalty = card_reject_penalty
        self.no_play_incorrect_penalty = no_play_incorrect_penalty
        self.no_play_correct_reduction = no_play_correct_reduction

    def setup_game(self) -> None:
        """Deal initial hands and place starter card."""
        # Shuffle deck
        self.state.deck.shuffle()

        # Deal cards to each scientist
        scientists = self.state.get_scientists()
        for _ in range(self.cards_per_scientist):
            for scientist in scientists:
                if not self.state.deck.is_empty():
                    scientist.hand.add_card(self.state.deck.draw())

        # Find and place starter card
        while not self.state.deck.is_empty():
            candidate = self.state.deck.draw()
            if self.rule.evaluate(candidate, []):
                self.state.mainline.add_card(candidate)
                logger.info(f"Starter card placed: {candidate}")
                break

    def evaluate_card(self, card: Card) -> bool:
        """Evaluate if a card is 'in' according to the rule."""
        mainline_cards = self.state.mainline.get_all()
        return self.rule.evaluate(card, mainline_cards)

    def play_turn(self, action: Action, advance_turn: bool = True) -> dict:
        """Process a player's action and update game state.

        Args:
            action: The action to process
            advance_turn: Whether to advance turn after processing (default: True)
        """
        current_player = self.state.get_current_player()
        result = {"player": current_player.name, "action": type(action).__name__}

        if isinstance(action, PlayCardAction):
            result.update(self._process_play_card(current_player, action))
        elif isinstance(action, NoPlayAction):
            result.update(self._process_no_play(current_player))
        elif isinstance(action, GuessRuleAction):
            result.update(self._process_guess(current_player, action))
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

        # Advance turn if game not over and advance_turn is True
        if not self.state.game_over and advance_turn:
            self.state.advance_turn()

        return result

    def _process_play_card(self, player: PlayerState, action: PlayCardAction) -> dict:
        """Process a card play action."""
        card = action.card

        if not player.hand.contains(card):
            return {"success": False, "reason": "Card not in hand"}

        # Remove card from hand
        player.hand.remove_card(card)

        # Evaluate card
        is_in = self.evaluate_card(card)

        if is_in:
            # Card accepted: add to mainline
            self.state.mainline.add_card(card)
            logger.info(f"{player.name} played {card} - ACCEPTED")
            return {
                "success": True,
                "accepted": True,
                "card": str(card),
                "can_guess": True,
            }
        else:
            # Card rejected: add to sideline, draw penalty cards
            self.state.add_sideline_card(card)
            drawn_cards = []
            for _ in range(self.card_reject_penalty):
                if not self.state.deck.is_empty():
                    drawn = self.state.deck.draw()
                    player.hand.add_card(drawn)
                    drawn_cards.append(str(drawn))

            if drawn_cards:
                cards_str = ", ".join(drawn_cards)
                logger.info(
                    f"{player.name} played {card} - REJECTED, "
                    f"drew {len(drawn_cards)} cards: {cards_str}"
                )
            else:
                logger.info(f"{player.name} played {card} - REJECTED, deck empty")

            return {
                "success": True,
                "accepted": False,
                "card": str(card),
                "can_guess": False,
            }

    def _process_no_play(self, player: PlayerState) -> dict:
        """Process a no-play declaration."""
        # Check if any card in hand would be accepted
        hand_cards = player.hand.get_all_cards()
        legal_cards = [c for c in hand_cards if self.evaluate_card(c)]

        if len(legal_cards) == 0:
            # Correct no-play: discard all cards, draw N-reduction (min 1)
            original_hand_size = len(hand_cards)

            # Discard all cards to sideline
            for card in hand_cards:
                player.hand.remove_card(card)
                self.state.add_sideline_card(card)

            # Draw new cards: N - reduction, but at least 1
            new_hand_size = max(1, original_hand_size - self.no_play_correct_reduction)
            drawn_cards = []
            for _ in range(new_hand_size):
                if not self.state.deck.is_empty():
                    drawn = self.state.deck.draw()
                    player.hand.add_card(drawn)
                    drawn_cards.append(str(drawn))

            logger.info(
                f"{player.name} correctly declared no-play, "
                f"discarded {original_hand_size} cards, drew {len(drawn_cards)} new cards"
            )

            return {
                "success": True,
                "correct": True,
                "can_guess": True,
            }
        else:
            # Incorrect no-play: rule-maker plays one legal card, draw penalty cards
            card_to_play = legal_cards[0]
            player.hand.remove_card(card_to_play)
            self.state.mainline.add_card(card_to_play)

            # Draw penalty cards
            drawn_cards = []
            for _ in range(self.no_play_incorrect_penalty):
                if not self.state.deck.is_empty():
                    drawn = self.state.deck.draw()
                    player.hand.add_card(drawn)
                    drawn_cards.append(str(drawn))

            if drawn_cards:
                logger.info(
                    f"{player.name} incorrectly declared no-play, "
                    f"{card_to_play} played, drew {len(drawn_cards)} penalty cards"
                )
            else:
                logger.info(
                    f"{player.name} incorrectly declared no-play, {card_to_play} played, deck empty"
                )

            return {
                "success": True,
                "correct": False,
                "forced_card": str(card_to_play),
                "can_guess": False,
            }

    def _process_guess(self, player: PlayerState, action: GuessRuleAction) -> dict:
        """Process a rule guess."""
        logger.info(f"{player.name} guessed: {action.guess_text}")

        # Log actual rule details
        logger.info("=" * 60)
        logger.info("ACTUAL RULE:")
        logger.info(f"Description: {self.rule.description()}")

        # Try to get Python code if it's a PythonRule
        from eleusis.python_rule import PythonRule
        if isinstance(self.rule, PythonRule):
            logger.info(f"Python code:\n{self.rule.get_code()}")
        else:
            logger.info("(No Python code - LLM-based rule)")

        logger.info("-" * 60)
        logger.info("GUESSED RULE:")
        logger.info(f"Description: {action.guess_text}")

        # Convert guessed rule to code for logging
        guessed_code = None
        if self.rule_validator and self.rule_validator.referee_client:
            try:
                guessed_code = self.rule_validator.referee_client.convert_rule_to_code(
                    action.guess_text
                )
                if guessed_code:
                    logger.info(f"Python code:\n{guessed_code}")
                else:
                    logger.warning("(Failed to convert guessed rule to Python code)")
            except Exception as e:
                logger.warning(f"(Could not convert guessed rule to code: {e})")

        logger.info("=" * 60)
        logger.info("")

        # Check if guess is correct using BOTH methods
        llm_correct = False
        llm_reasoning = ""
        sim_correct = False
        sim_reasoning = ""
        sim_comparisons = 0
        sim_mismatches = 0

        if self.rule_validator:
            # Method 1: LLM-based comparison
            try:
                secret_rule_text = self.rule.description()
                llm_correct, llm_reasoning = self.rule_validator.check_equivalence(
                    secret_rule_text, action.guess_text, self.state.mainline.to_str()
                )
                logger.info(f"LLM verdict: {llm_correct} - {llm_reasoning}")
            except Exception as e:
                logger.error(f"Failed LLM equivalence check: {e}")
                llm_correct = False
                llm_reasoning = f"Error checking equivalence: {e}"

            # Method 2: Simulation-based comparison
            # Pass the already-converted code to avoid re-conversion
            try:
                (
                    sim_correct,
                    sim_reasoning,
                    sim_comparisons,
                    sim_mismatches,
                ) = self.rule_validator.check_equivalence_by_simulation(
                    self.rule,
                    action.guess_text,
                    self.state.mainline.get_all(),
                    preconverted_code=guessed_code,  # Reuse the code we already converted
                )
                logger.info(
                    f"Simulation verdict: {sim_correct} - {sim_reasoning} "
                    f"({sim_comparisons} comparisons, {sim_mismatches} mismatches)"
                )
            except Exception as e:
                logger.error(f"Failed simulation equivalence check: {e}")
                sim_correct = False
                sim_reasoning = f"Error in simulation: {e}"

            # Check if verdicts differ
            if llm_correct != sim_correct:
                logger.warning(
                    f"⚠️  VERDICT MISMATCH: LLM says {llm_correct}, "
                    f"Simulation says {sim_correct}"
                )
                logger.warning(f"  LLM reasoning: {llm_reasoning}")
                logger.warning(f"  Simulation reasoning: {sim_reasoning}")

        # Use simulation verdict as final decision (more reliable)
        is_correct = sim_correct
        reasoning = sim_reasoning

        if is_correct:
            # Correct guess! Mark game state
            self.rule_guessed = True
            self.winning_guesser = player.name
            logger.info(f"{player.name} correctly guessed the rule!")
            return {
                "success": True,
                "correct": True,
                "guess": action.guess_text,
                "reasoning": reasoning,
                "llm_verdict": llm_correct,
                "sim_verdict": sim_correct,
                "sim_comparisons": sim_comparisons,
            }
        else:
            # Incorrect guess - draw penalty card
            if not self.state.deck.is_empty():
                drawn = self.state.deck.draw()
                player.hand.add_card(drawn)
                logger.info(f"{player.name} incorrect guess, drew {drawn}")

            return {
                "success": True,
                "correct": False,
                "guess": action.guess_text,
                "reasoning": reasoning,
                "llm_verdict": llm_correct,
                "sim_verdict": sim_correct,
                "sim_comparisons": sim_comparisons,
            }

    def check_mandatory_guess(self, player: PlayerState) -> bool:
        """Check if player must guess (hand size 0)."""
        return player.hand.size() == 0

    def is_game_over(self) -> bool:
        """Check if game should end."""
        # Game ends if rule guessed correctly
        if self.rule_guessed:
            return True

        # Game ends if deck empty and no one can make a legal play
        if self.state.deck.is_empty():
            scientists = self.state.get_scientists()
            for scientist in scientists:
                hand_cards = scientist.hand.get_all_cards()
                if any(self.evaluate_card(c) for c in hand_cards):
                    return False
            return True

        return False

    def calculate_scores(self) -> dict[str, int]:
        """Calculate final scores for the round."""
        scores = {}

        # Scientists score based on remaining cards
        scientists = self.state.get_scientists()
        scientist_scores = []

        for scientist in scientists:
            score = scientist.hand.size()
            # Apply bonus if they guessed correctly
            if self.rule_guessed and scientist.name == self.winning_guesser:
                score += self.correct_guess_bonus
            scores[scientist.name] = score
            scientist_scores.append(score)

        # Rule-maker gets second-lowest scientist score
        scientist_scores.sort()
        rule_maker_score = (
            scientist_scores[1] if len(scientist_scores) >= 2 else scientist_scores[0]
        )
        rule_maker = self.state.get_rule_maker()
        scores[rule_maker.name] = rule_maker_score

        logger.info(f"Round scores: {scores}")
        return scores

    def end_game(self, winner_name: str | None = None) -> None:
        """Mark game as over and record winner."""
        self.state.game_over = True
        self.state.winner = winner_name
        logger.info(f"Game ended. Winner: {winner_name}")
