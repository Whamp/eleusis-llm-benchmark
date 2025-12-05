"""Game engine for Eleusis: turn logic, game flow, and scoring."""

import logging
import random
import textwrap
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


class GameEngine:
    """Orchestrates Eleusis game progression."""

    def __init__(
        self,
        game_state: GameState,
        rule: Rule,
        game_master,
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
        self.game_master = game_master
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

        # Deal cards to each player
        for _ in range(self.cards_per_scientist):
            for player in self.state.players:
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

        # Special case: empty hand (player reached 0 cards)
        if len(hand_cards) == 0:
            logger.warning(f"{player.name} declared no-play with empty hand")
            return {"success": True, "correct": True, "can_guess": True}

        legal_cards = [c for c in hand_cards if self.evaluate_card(c)]

        if len(legal_cards) == 0:
            # Correct no-play: place one random card in sideline, discard rest, draw N-reduction (min 1)
            original_hand_size = len(hand_cards)

            # Choose one random card to place in sideline
            sideline_card = random.choice(hand_cards)
            player.hand.remove_card(sideline_card)
            self.state.add_sideline_card(sideline_card)

            # Discard remaining cards from hand (not to sideline)
            for card in list(player.hand.get_all_cards()):
                player.hand.remove_card(card)

            # Draw new cards: N - reduction, can reach 0
            new_hand_size = max(0, original_hand_size - self.no_play_correct_reduction)
            drawn_cards = []
            for _ in range(new_hand_size):
                if not self.state.deck.is_empty():
                    drawn = self.state.deck.draw()
                    player.hand.add_card(drawn)
                    drawn_cards.append(str(drawn))

            logger.info(
                f"{player.name} correctly declared no-play, "
                f"placed {sideline_card} in sideline, discarded {original_hand_size - 1} cards, "
                f"drew {len(drawn_cards)} new cards"
            )

            return {"success": True, "correct": True, "can_guess": True,}

        else:
            # Incorrect no-play: game master randomly plays one legal card, draw penalty cards
            card_to_play = random.choice(legal_cards)
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

            return {"success": True, "correct": False, "forced_card": str(card_to_play), "can_guess": False,}


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

        # Log both verdicts for debugging
        logger.info(f"Simulation verdict: {is_correct}")

        # Record failed guess
        if not is_correct:
            self.state.record_failed_guess(player_name=player.name, guess_text=action.guess_text)

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
                **metadata,
            }
        else:
            # Incorrect guess - draw penalty card
            logger.info(f"{player.name} incorrect guess")

            if not self.state.deck.is_empty():
                drawn = self.state.deck.draw()
                player.hand.add_card(drawn)
                logger.info(f"{player.name} drew penalty card: {drawn}")

            return {
                "success": True,
                "correct": False,
                "guess": action.guess_text,
                "reasoning": reasoning,
                **metadata,
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
            for player in self.state.players:
                hand_cards = player.hand.get_all_cards()
                if any(self.evaluate_card(c) for c in hand_cards):
                    return False
            return True

        return False

    def calculate_scores(self) -> dict[str, int]:
        """Calculate final scores for the round."""
        scores = {}

        # Players score based on remaining cards
        for player in self.state.players:
            score = player.hand.size()
            # Apply bonus if they guessed correctly
            if self.rule_guessed and player.name == self.winning_guesser:
                score += self.correct_guess_bonus
            scores[player.name] = score

        logger.info(f"Round scores: {scores}")
        return scores

    def end_game(self, winner_name: str | None = None) -> None:
        """Mark game as over and record winner."""
        self.state.game_over = True
        self.state.winner = winner_name
        logger.info(f"Game ended. Winner: {winner_name}")
