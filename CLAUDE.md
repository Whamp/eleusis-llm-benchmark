# CLAUDE.md for Eleusis LLM Benchmark

## Project Overview

This is an LLM benchmark for the card game Eleusis. 
The benchmark evaluates how well LLMs can deduce secret rules by playing as "Scientists" who test cards against a hidden rule.

Key concepts:
- **Mainline**: Horizontal row of accepted cards, visible to all players
- **Sideline**: Columns of rejected cards beneath mainline cards
- **Rule**: A deterministic Python function that evaluates cards based only on the card and mainline state
- **Scientists**: LLM players who try to deduce the rule through experimentation
- **Rule-maker/Game Master**: LLM that generates rules and judges equivalence


```bash
# Generate a library of rules
uv run scripts/generate_rule_library.py --num-rules 50 --output rules.json --model openai/gpt-oss-120b --max-tokens 16384 --test-cases 5

# Evaluate rules in library
uv run scripts/evaluate_rules.py --library rules.json

# Play a single round with default config
uv run scripts/play_single_round.py

# Play a tournament (multiple rounds)
uv run scripts/play_tournament.py
```

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
- Comparison: Uses simulation (authoritative) to check rule equivalence by playing multiple simulated games
- LLM comparison also available for debugging

**game_runner.py** - High-level orchestrator that sets up and runs complete rounds

**player.py**
- `LLMScientist`: LLM-powered player that selects cards/actions and optionally guesses rules
- `RandomScientist`: Random baseline player for testing

**game_master.py** - LLM-powered game master that:
- Generates rules from natural language descriptions
- Converts rule descriptions to executable Python code
- Judges rule equivalence using LLM reasoning

**llm_client.py** - `HuggingFaceClient` wraps Hugging Face Inference Providers API:
- Handles API calls with retry logic
- Extracts structured responses from XML tags or code blocks
- Supports response continuation when truncated
- Methods for rule equivalence, code conversion, card evaluation

**prompts.py** - Prompt templates for LLM interactions (move selection, rule generation, comparison, etc.)

### Key Workflows

**Rule Generation**: Game master generates natural language rule → converts to Python code → validates with test cases

**Turn Processing**: Player selects action → GameEngine processes (evaluate card, update state, apply penalties/bonuses) → optionally allow rule guess → advance turn

**Rule Comparison**: Player guesses rule → Game master converts guess to code → RuleValidator runs simulations comparing actual vs guessed → returns verdict (authoritative) + LLM verdict (debugging)

### Configuration

**config.yaml** - Controls all game parameters:
- Model selection for game master and scientists (Hugging Face model names)
- Rule source (LLM-generated or pre-generated library)
- Game parameters (hand size, penalties, bonuses, max turns)
- Tournament settings

**HF_TOKEN** environment variable required for Hugging Face API access (use .env file)

## Development Notes

- Rules are compiled into sandboxed Python functions with limited builtins (len, sum, min, max, any, all)
- Rule code must be function body only, not full function definition
- Simulation-based rule comparison is authoritative; LLM comparison is for debugging only
- Failed guesses are tracked to prevent duplicates
- Logging: Uses Python logging module with separate console/file levels
- Game state uses compact string representation for efficient LLM prompting
