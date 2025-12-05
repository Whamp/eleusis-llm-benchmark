"""Solo game engine for single-player pattern discovery card game."""

import logging
import textwrap
from dataclasses import dataclass

from eleusis.cards import Card
from eleusis.game_state import GameState, PlayerState

logger = logging.getLogger(__name__)


# Action types for solo mode (GuessRuleAction imported as-is)
@dataclass
class PlayCardAction:
    """Action to play a card from hand."""

    card: Card


@dataclass
class GuessRuleAction:
    """Action to guess the secret rule."""

    guess_text: str


Action = PlayCardAction | GuessRuleAction


class Rule:
    """A game rule with description and executable Python code."""

    def __init__(self, description: str, code: str):
        """Initialize rule and compile code."""
        self.description_text = description
        self.code = code
        self._eval_function = self._compile_code(code)

    def _compile_code(self, code: str):
        """Compile Python code into executable function."""

        # Validate code structure
        first_line = code.strip().split('\n')[0].strip() if code.strip() else ""
        if first_line.startswith('def '):
            logger.error("Generated code appears to be a complete function definition!")
            logger.error(f"First line: {first_line}")
            logger.error("Expected function body only, not 'def ...'.")

        if 'return' not in code:
            logger.warning("Generated code has no 'return' statement")

        # Debug print function
        def debug_print(*args):
            """Print function for debugging rule execution."""
            logger.debug(f"  [Rule Debug] {' '.join(str(arg) for arg in args)}")

        # Safe execution environment
        safe_globals = {
            "__builtins__": {
                "len": len,
                "sum": sum,
                "abs": abs,
                "min": min,
                "max": max,
                "any": any,
                "all": all,
                "print": debug_print,
            },
            "Card": Card,
        }

        # Wrap code in function definition
        full_code = f"""
def evaluate_rule(card, mainline):
{textwrap.indent(code, '    ')}
"""

        # Execute code to define the function (fails hard on syntax errors)
        local_namespace = {}
        exec(full_code, safe_globals, local_namespace)

        return local_namespace["evaluate_rule"]

    def evaluate(self, card: Card, mainline: list[Card]) -> bool:
        """Evaluate if card is accepted according to rule."""

        # Call eval function (fails hard on runtime errors)
        result = self._eval_function(card, mainline)

        logger.debug(f"Evaluating {card} for mainline {mainline} : Result: {bool(result)} for rule '{self.description_text[:100]}...'")
        return bool(result)

    def description(self) -> str:
        """Return rule description."""
        return self.description_text

    def get_code(self) -> str:
        """Return rule code."""
        return self.code


class GameEngineSolo:
    """Simplified game engine for solo pattern discovery mode."""

    def __init__(
        self,
        game_state: GameState,
        rule: Rule,
        game_master,
        rule_validator=None,
        hand_size: int = 12,
        wrong_guess_penalty: int = 3,
    ) -> None:
        """Initialize solo game engine with state and rule."""
        self.state = game_state
        self.rule = rule
        self.game_master = game_master
        self.rule_validator = rule_validator
        self.rule_guessed = False
        self.winning_turn: int | None = None
        self.failed_guess_count = 0
        self.hand_size = hand_size
        self.wrong_guess_penalty = wrong_guess_penalty

    def setup_game(self) -> None:
        """Deal initial hand and place starter card."""
        # Shuffle deck
        self.state.deck.shuffle()

        # Deal cards to the single player
        player = self.state.players[0]
        for _ in range(self.hand_size):
            if not self.state.deck.is_empty():
                player.hand.add_card(self.state.deck.draw())

        # Find and place starter card
        while not self.state.deck.is_empty():
            candidate = self.state.deck.draw()
            if self.rule.evaluate(candidate, []):
                self.state.mainline.add_card(candidate)
                logger.info(f"Starter card placed: {candidate}")
                break

    def evaluate_card(self, card: Card) -> bool:
        """Evaluate if a card is accepted according to the rule."""
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
        elif isinstance(action, GuessRuleAction):
            result.update(self._process_guess(current_player, action))
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

        # Advance turn if game not over and advance_turn is True
        if not self.state.game_over and advance_turn:
            self.state.advance_turn()

        return result

    def _process_play_card(self, player: PlayerState, action: PlayCardAction) -> dict:
        """Process a card play action with constant hand size."""
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
        else:
            # Card rejected: add to sideline
            self.state.add_sideline_card(card)
            logger.info(f"{player.name} played {card} - REJECTED")

        # Always draw 1 card to maintain constant hand size
        if not self.state.deck.is_empty():
            drawn = self.state.deck.draw()
            player.hand.add_card(drawn)
            drawn_str = str(drawn)
            logger.info(f"{player.name} drew 1 card: {drawn_str}")
        else:
            logger.info(f"{player.name} could not draw (deck empty)")

        return {
            "success": True,
            "accepted": is_in,
            "card": str(card),
        }

    def _process_guess(self, player: PlayerState, action: GuessRuleAction) -> dict:
        """Process rule guess with simulation-based comparison."""
        logger.info(f"{player.name} guessed: {action.guess_text}")

        # Log actual rule details
        logger.info("-" * 60)
        logger.info("ACTUAL RULE:")
        logger.info(f"Description: {self.rule.description()}")
        logger.info(f"Python code:\n{self.rule.get_code()}")

        logger.info("-" * 60)
        logger.info("GUESSED RULE:")
        logger.info(f"Description: {action.guess_text}")
        logger.info("-" * 60)
        logger.info("")

        # Compare rules using RuleValidator
        is_correct, reasoning, metadata = self.rule_validator.compare_rules(
            actual_rule=self.rule,
            guessed_rule_desc=action.guess_text,
            current_mainline=self.state.mainline.get_all(),
            game_master=self.game_master,
            num_simulations=2,
            turns_per_simulation=10,
        )

        # Log verdict
        logger.info(f"Simulation verdict: {is_correct}")

        # Record failed guess
        if not is_correct:
            self.state.record_failed_guess(player_name=player.name, guess_text=action.guess_text)
            self.failed_guess_count += 1

        if is_correct:
            # Correct guess! Mark game state
            self.rule_guessed = True
            self.winning_turn = self.state.current_turn_index
            logger.info(f"{player.name} correctly guessed the rule!")
            return {
                "success": True,
                "correct": True,
                "guess": action.guess_text,
                "reasoning": reasoning,
                **metadata,
            }
        else:
            # Incorrect guess
            logger.info(f"{player.name} incorrect guess (penalty applied)")

            return {
                "success": True,
                "correct": False,
                "guess": action.guess_text,
                "reasoning": reasoning,
                **metadata,
            }

    def is_game_over(self) -> bool:
        """Check if game should end."""
        # Game ends if rule guessed correctly
        return self.rule_guessed

    def calculate_scores(self, max_turns: int, current_turn: int) -> dict[str, int]:
        """Calculate final score for solo mode.

        Score = max_turns - current_turn - (penalty * failed_guesses) if correct guess
        Score = 0 if no correct guess
        """
        scores = {}
        player = self.state.players[0]

        if self.rule_guessed:
            # Score based on efficiency: fewer turns and fewer failed guesses = higher score
            score = max_turns - current_turn - (self.wrong_guess_penalty * self.failed_guess_count)
            scores[player.name] = score
            logger.info(
                f"Solo score: {score} "
                f"(max_turns={max_turns}, current_turn={current_turn}, "
                f"failed_guesses={self.failed_guess_count}, penalty={self.wrong_guess_penalty})"
            )
        else:
            # No correct guess = score of 0
            scores[player.name] = 0
            logger.info(f"Solo score: 0 (no correct guess)")

        return scores

    def end_game(self, winner_name: str | None = None) -> None:
        """Mark game as over and record winner."""
        self.state.game_over = True
        self.state.winner = winner_name
        logger.info(f"Game ended. Winner: {winner_name}")
