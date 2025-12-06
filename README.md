# Eleusis LLM Benchmark

This project is a benchmark for evaluating large language models (LLMs) on their ability to understand and generate rules in the context of the card game Eleusis.

The core game is implemented in Python, following the provided simplified ruleset, see RULES.md for details.

A testing framework is provided to facilitate the assessment of LLMs' performance in playing the game.

Open source LLM are called using Hugging Face Inference Providers API.

Closed source LLMs are called using their respective APIs.

```bash
# Generate a library of rules
uv run scripts/generate_rule_library.py --num-rules 50 --output rules.json

# Evaluate rules in library
uv run scripts/evaluate_rules.py --library rules.json

# Multi-player tournament mode
uv run scripts/play_single_round.py  # Single round
uv run scripts/play_tournament.py    # Multiple rounds
uv run scripts/tournament_analysis.py results/tournament_results_TIMESTAMP.json

# Solo mode (single-player pattern discovery)
uv run scripts/evaluate_single.py
```


## Project Overview

This is an LLM benchmark for pattern discovery in card games. The benchmark evaluates how well LLMs can deduce secret rules by testing cards against a hidden pattern.

The project supports two modes:

### Tournament Mode (Multi-player)
Multiple LLM "Scientists" compete to deduce a secret rule while managing their card hands. Based on the card game Eleusis with full game mechanics including hand management, penalties, and scoring.

### Solo Mode (Single-player)
A single LLM player attempts to discover the pattern as efficiently as possible. Simplified mechanics with constant hand size, focusing on pattern discovery speed and accuracy.

Key concepts:
- **Mainline**: Horizontal row of accepted cards, visible to all players
- **Sideline**: Columns of rejected cards beneath mainline cards
- **Rule**: A deterministic Python function that evaluates cards based only on the card and mainline state
- **Scientists/Player**: LLM players who try to deduce the rule through experimentation
- **Game Master**: LLM that generates rules and judges equivalence


## Architecture

### Core Game Components (src/eleusis/)

**cards.py** - Card representation with rank (1-13) and suit (Hearts, Diamonds, Clubs, Spades)

**game_state.py** - Game state including mainline, sidelines, player hands, deck, turn tracking

**game_engine.py**
- `Rule` class: Wraps rule description and executable Python code. Compiles natural language rule into executable function that evaluates cards
- `GameEngine`: Orchestrates game flow, processes actions (PlayCardAction, NoPlayAction, GuessRuleAction), manages scoring
- Action types define player moves

**rules.py** - `RuleValidator` validates rules and compares guessed rules to actual rules:
- Validation: Tests determinism, empty mainline handling, random scenarios
- Comparison: Uses simulation-based comparison to check rule equivalence by playing multiple simulated games

**game_runner.py** - High-level orchestrator that sets up and runs complete rounds

**player.py**
- `LLMScientist`: LLM-powered player that selects cards/actions and optionally guesses rules
- `RandomScientist`: Random baseline player for testing

**game_master.py** - LLM-powered game master that:
- Converts rule descriptions to executable Python code

**llm_client.py** - `HuggingFaceClient` wraps Hugging Face Inference Providers API:
- Handles API calls with retry logic
- Extracts structured responses from XML tags or code blocks
- Supports response continuation when truncated
- Methods for code conversion and card evaluation

**prompts.py** - Prompt templates for LLM interactions (move selection, rule generation for library, etc.)

### Key Workflows

**Rule Loading**: Rules are loaded from a pre-generated library (JSON file)

**Turn Processing**: Player selects action → GameEngine processes (evaluate card, update state, apply penalties/bonuses) → optionally allow rule guess → advance turn

**Rule Comparison**: Player guesses rule → Game master converts guess to code → RuleValidator runs simulations comparing actual vs guessed → returns verdict

### Configuration

**config.yaml** - Controls all game parameters:
- Model selection for game master and scientists (Hugging Face model names)
- Rule library path and selection mode (random or sequential)
- Game parameters (hand size, penalties, bonuses, max turns)
- Tournament settings

**HF_TOKEN** environment variable required for Hugging Face API access (use .env file)

### Solo Mode Details

Solo mode is a simplified single-player variant focused on pattern discovery efficiency:

**Game Mechanics:**
- Constant hand size (12 cards by default) - always draw 1 card after playing
- No "no play" action - simply play a card each turn
- Can guess the rule at any time (not restricted to successful plays)
- Game ends when rule is guessed correctly or max turns reached (40 by default)

**Scoring:**
- If correct guess: `score = max_turns - current_turn - (3 × failed_guesses)`
- If no correct guess: `score = 0`
- Higher scores are better (reward efficiency: fewer turns, fewer wrong guesses)

**Configuration:**
Edit `config.yaml` under the `solo_game` section to configure:
- Player model and temperature
- Hand size (constant throughout game)
- Max turns per round
- Wrong guess penalty (default: 3 points per failed guess)
- Number of evaluation rounds

**Implementation:**
- `game_engine_solo.py`: Simplified game engine for solo mode
- `game_runner_solo.py`: Solo round orchestration
- `prompts_solo.py`: Pattern discovery prompts (no "Eleusis" references)
- `evaluate_single.py`: Multi-round evaluation script

**Resume Interrupted Evaluations:**

If an evaluation is interrupted (crash, Ctrl-C, etc.), you can resume from where it left off:

```bash
uv run scripts/evaluate_single.py --resume results/solo_evaluation_20251205_151306
```

**Requirements for resume:**
- Works only with `selection: "sequential"` in config.yaml (deterministic rule order)
- The results folder must contain a valid results.json file with checkpoint data
- Configuration parameters (num_rounds, max_turns, hand_size, etc.) must match the original run

**What happens on resume:**
- Loads checkpoint state from results.json (rules, progress, statistics)
- Validates configuration consistency with original run
- Resumes from the next incomplete round
- Continues using rules from the stored checkpoint (self-contained, no dependency on external rules.json)
- Appends new results to the same results.json file

**Checkpoint data:**
The resume feature stores complete rule information in results.json, making checkpoints self-contained. Even if the rules.json file changes or is deleted, resume will work using the stored rules.

## Development Notes

- Rules are compiled into sandboxed Python functions with limited builtins (len, sum, min, max, any, all)
- Rule code must be function body only, not full function definition
- Simulation-based rule comparison is used to validate guesses
- Failed guesses are tracked to prevent duplicates
- Logging: Uses Python logging module with separate console/file levels
- Game state uses compact string representation for efficient LLM prompting