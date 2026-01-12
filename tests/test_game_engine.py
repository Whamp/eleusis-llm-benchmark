"""Unit tests for game engine module."""

from unittest.mock import Mock

from eleusis.game import (
    Card,
    GameEngine,
    GameState,
    GuessRuleAction,
    PlayCardAction,
    Rule,
    Suit,
)


def AlwaysAcceptRule() -> Rule:
    """Test rule that accepts all cards."""
    return Rule("Accept all cards", "return True")


def EvenRankRule() -> Rule:
    """Test rule that accepts only even ranks."""
    return Rule("Only even ranks are accepted", "return card.rank % 2 == 0")


class TestGameEngine:
    """Tests for GameEngine class."""

    def test_game_setup(self) -> None:
        """Test game setup with dealing and starter card."""
        state = GameState("Player")
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule, rule_compiler_client=None, hand_size=12)

        engine.setup_game()

        # Check player has 12 cards
        assert state.player.hand.size() == 12

        # Check starter card is placed
        assert state.mainline.size() == 1

    def test_play_card_accepted(self) -> None:
        """Test playing a card that is accepted."""
        state = GameState("Player")
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule, rule_compiler_client=None, hand_size=12)
        engine.setup_game()

        card = state.player.hand.get_all_cards()[0]

        result = engine.play_turn(PlayCardAction(card))

        assert result["success"]
        assert result["accepted"]
        # Constant hand size: after playing, draw 1 card
        assert state.player.hand.size() == 12
        assert state.mainline.size() == 2  # Starter + played card

    def test_play_card_rejected(self) -> None:
        """Test playing a card that is rejected."""
        state = GameState("Player")
        rule = EvenRankRule()
        engine = GameEngine(state, rule, rule_compiler_client=None, hand_size=12)

        # Manually set up for testing
        state.mainline.add_card(Card(2, Suit.HEARTS))  # Starter
        odd_card = Card(3, Suit.DIAMONDS)
        state.player.hand.add_card(odd_card)

        result = engine.play_turn(PlayCardAction(odd_card))

        assert result["success"]
        assert not result["accepted"]
        assert state.mainline.size() == 1  # Only starter
        assert 0 in state.sidelines  # Card in sideline

    def test_calculate_score_with_guess(self) -> None:
        """Test score calculation (efficiency-based)."""
        state = GameState("Player")
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule, rule_compiler_client=None, hand_size=12, wrong_guess_penalty=3)

        # Simulate a correct guess at turn 10
        engine.rule_guessed = True
        engine.failed_guess_count = 0

        score = engine.calculate_score(max_turns=40, current_turn=10)

        # Score = max_turns - current_turn - (penalty * failed_guesses)
        # Score = 40 - 10 - 0 = 30
        assert score == 30

    def test_calculate_score_with_failed_guesses(self) -> None:
        """Test score calculation with failed guesses."""
        state = GameState("Player")
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule, rule_compiler_client=None, hand_size=12, wrong_guess_penalty=3)

        # Simulate a correct guess at turn 15 with 2 failed guesses
        engine.rule_guessed = True
        engine.failed_guess_count = 2

        score = engine.calculate_score(max_turns=40, current_turn=15)

        # Score = max_turns - current_turn - (penalty * failed_guesses)
        # Score = 40 - 15 - (3 * 2) = 40 - 15 - 6 = 19
        assert score == 19

    def test_calculate_score_no_correct_guess(self) -> None:
        """Test score calculation when rule was not guessed."""
        state = GameState("Player")
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule, rule_compiler_client=None, hand_size=12, wrong_guess_penalty=3)

        # No correct guess
        engine.rule_guessed = False
        engine.failed_guess_count = 3

        score = engine.calculate_score(max_turns=40, current_turn=40)

        # No correct guess = score of 0
        assert score == 0

    def test_is_game_over_after_correct_guess(self) -> None:
        """Test game over when rule is guessed correctly."""
        state = GameState("Player")
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule, rule_compiler_client=None, hand_size=12)

        # Initially not over
        assert not engine.is_game_over()

        # After correct guess
        engine.rule_guessed = True
        assert engine.is_game_over()

    def test_play_turn_returns_result(self) -> None:
        """Test that play_turn returns appropriate result dict."""
        state = GameState("Player")
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule, rule_compiler_client=None, hand_size=12)
        engine.setup_game()

        card = state.player.hand.get_all_cards()[0]

        result = engine.play_turn(PlayCardAction(card))

        # Result should contain expected keys
        assert "success" in result
        assert "accepted" in result
        assert result["success"] is True

    def test_constant_hand_size(self) -> None:
        """Test that hand size remains constant after each play."""
        state = GameState("Player")
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule, rule_compiler_client=None, hand_size=12)
        engine.setup_game()

        # Play multiple cards, hand size should stay at 12
        for _ in range(5):
            assert state.player.hand.size() == 12
            card = state.player.hand.get_all_cards()[0]
            engine.play_turn(PlayCardAction(card))

    def test_process_guess_correct(self) -> None:
        """Test processing a correct rule guess."""
        state = GameState("Player")
        rule = AlwaysAcceptRule()

        # Mock rule validator that returns correct guess
        mock_validator = Mock()
        mock_validator.compare_rules.return_value = (
            True,  # is_correct
            "Rules match",  # reasoning
            {"simulation_comparisons": 100, "simulation_mismatches": 0}  # metadata
        )

        # Mock game master (LLM client)
        mock_game_master = Mock()

        engine = GameEngine(
            state, rule,
            rule_compiler_client=mock_game_master,
            rule_validator=mock_validator,
            hand_size=12
        )
        engine.setup_game()

        # Make a guess
        result = engine.play_turn(GuessRuleAction("Accept all cards"))

        assert result["success"]
        assert result["correct"]
        assert engine.rule_guessed
        assert engine.failed_guess_count == 0

    def test_process_guess_incorrect(self) -> None:
        """Test processing an incorrect rule guess."""
        state = GameState("Player")
        rule = AlwaysAcceptRule()

        # Mock rule validator that returns incorrect guess
        mock_validator = Mock()
        mock_validator.compare_rules.return_value = (
            False,  # is_correct
            "Rules differ at card X",  # reasoning
            {"simulation_comparisons": 50, "simulation_mismatches": 1}  # metadata
        )

        mock_game_master = Mock()

        engine = GameEngine(
            state, rule,
            rule_compiler_client=mock_game_master,
            rule_validator=mock_validator,
            hand_size=12,
            wrong_guess_penalty=3
        )
        engine.setup_game()

        # Make a guess
        result = engine.play_turn(GuessRuleAction("Wrong guess"))

        assert result["success"]
        assert not result["correct"]
        assert not engine.rule_guessed
        assert engine.failed_guess_count == 1

        # Failed guess should be recorded
        assert len(state.failed_rule_guesses) == 1
        assert state.failed_rule_guesses[0]["guess"] == "Wrong guess"

    def test_multiple_failed_guesses(self) -> None:
        """Test that multiple failed guesses accumulate correctly."""
        state = GameState("Player")
        rule = AlwaysAcceptRule()

        mock_validator = Mock()
        mock_validator.compare_rules.return_value = (False, "Wrong", {})
        mock_game_master = Mock()

        engine = GameEngine(
            state, rule,
            rule_compiler_client=mock_game_master,
            rule_validator=mock_validator,
            hand_size=12,
            wrong_guess_penalty=3
        )
        engine.setup_game()

        # Make multiple wrong guesses
        engine.play_turn(GuessRuleAction("Guess 1"))
        engine.play_turn(GuessRuleAction("Guess 2"))
        engine.play_turn(GuessRuleAction("Guess 3"))

        assert engine.failed_guess_count == 3
        assert len(state.failed_rule_guesses) == 3

        # Check score with penalty
        score = engine.calculate_score(max_turns=40, current_turn=10)
        # If not guessed correctly, score is 0
        assert score == 0
