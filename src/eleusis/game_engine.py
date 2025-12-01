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
        self, game_state: GameState, rule: Rule, rule_validator=None
    ) -> None:
        """Initialize game engine with state and rule."""
        self.state = game_state
        self.rule = rule
        self.rule_validator = rule_validator
        self.rule_guessed = False
        self.winning_guesser: str | None = None

    def setup_game(self) -> None:
        """Deal initial hands and place starter card."""
        # Shuffle deck
        self.state.deck.shuffle()

        # Deal 12 cards to each scientist
        scientists = self.state.get_scientists()
        for _ in range(12):
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

    def play_turn(self, action: Action) -> dict:
        """Process a player's action and update game state."""
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

        # Advance turn if game not over
        if not self.state.game_over:
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
            # Card rejected: add to sideline, draw card
            self.state.add_sideline_card(card)
            if not self.state.deck.is_empty():
                drawn = self.state.deck.draw()
                player.hand.add_card(drawn)
                logger.info(f"{player.name} played {card} - REJECTED, drew {drawn}")
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
            # Correct no-play: choose one card to discard
            if hand_cards:
                card_to_discard = hand_cards[0]
                player.hand.remove_card(card_to_discard)
                self.state.add_sideline_card(card_to_discard)
                logger.info(
                    f"{player.name} correctly declared no-play, discarded {card_to_discard}"
                )

            return {
                "success": True,
                "correct": True,
                "can_guess": True,
            }
        else:
            # Incorrect no-play: rule-maker plays one legal card
            card_to_play = legal_cards[0]
            player.hand.remove_card(card_to_play)
            self.state.mainline.add_card(card_to_play)

            # Draw penalty card
            if not self.state.deck.is_empty():
                drawn = self.state.deck.draw()
                player.hand.add_card(drawn)
                logger.info(
                    f"{player.name} incorrectly declared no-play, "
                    f"{card_to_play} played, drew {drawn}"
                )
            else:
                logger.info(f"{player.name} incorrectly declared no-play, {card_to_play} played")

            return {
                "success": True,
                "correct": False,
                "forced_card": str(card_to_play),
                "can_guess": False,
            }

    def _process_guess(self, player: PlayerState, action: GuessRuleAction) -> dict:
        """Process a rule guess."""
        logger.info(f"{player.name} guessed: {action.guess_text}")

        # Check if guess is correct
        is_correct = False
        reasoning = ""

        if self.rule_validator:
            try:
                secret_rule_text = self.rule.description()
                is_correct, reasoning = self.rule_validator.check_equivalence(
                    secret_rule_text, action.guess_text
                )
                logger.info(f"Referee verdict: {is_correct} - {reasoning}")
            except Exception as e:
                logger.error(f"Failed to check rule equivalence: {e}")
                is_correct = False
                reasoning = "Error checking equivalence"

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
            # Subtract 3 if they guessed correctly
            if self.rule_guessed and scientist.name == self.winning_guesser:
                score -= 3
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
