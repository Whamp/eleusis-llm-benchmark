# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See README.md for project overview, usage instructions and architecture details.

Please also carefully read the global CLAUDE.md file, containing important guidelines that apply to all projects including this one.

## Common Commands

```bash
# Install dependencies
uv sync

# Run solo evaluation (single model)
uv run scripts/evaluate_single.py

# Run with specific model and parameters
uv run scripts/evaluate_single.py --player "openrouter:anthropic/claude-3.5-haiku" --num-rounds 10 --tag test

# Run parallel evaluation (multiple models)
./scripts/run_parallel_eval.sh

# Run tests
uv run pytest

# Run single test file
uv run pytest tests/test_game_engine.py

# Run specific test
uv run pytest tests/test_game_engine.py::TestGameEngineSolo::test_game_setup

# Generate new rules
uv run scripts/generate_rule_library.py --num-rules 50 --output rules.json

# Evaluate rule acceptance rates
uv run scripts/evaluate_rules.py --library rules.json

# Lint
uv run ruff check src/ scripts/ tests/
```

## Architecture

### Game Flow

1. **Rule Loading**: `RuleFactory` loads rules from `rules.json` library, filtering by acceptance rate
2. **Rule Compilation**: Rules are Python function bodies (no `def`) compiled into sandboxed executables with limited builtins
3. **Game Setup**: `GameEngineSolo` deals cards, places starter card that passes the rule
4. **Game Loop**: `LLMScientistSolo` plays cards and optionally guesses the rule each turn
5. **Rule Validation**: `RuleValidator.compare_rules()` uses simulation-based comparison to verify guesses

### Key Design Decisions

- **Rule Format**: Rules are function bodies only, wrapped with `def evaluate_rule(card, mainline):`. They have access to `card.rank`, `card.color`, `card.suit.suit_name`, and `mainline` (list of previous cards).
- **Constant Hand Size**: Player always has 12 cards (draws 1 after each play)
- **Model Specification**: Prefix-based routing (`openrouter:model-name` or `hf:model-name`)
- **Simulation Equivalence**: Rule guesses are converted to code and compared against actual rule by running both on all 52 cards across multiple simulated turns
- **Checkpointing**: Evaluations save after each round and can resume with `--resume`

### Provider System

`create_client(model_spec)` in `src/eleusis/providers/__init__.py` routes to:
- `OpenRouterClient` for `openrouter:` prefix
- `HuggingFaceClient` for `hf:` prefix or no prefix

### Configuration

`config.yaml` controls:
- Provider API keys (via env vars)
- Game master model (for rule compilation)
- Rule source and filtering
- Game parameters (rounds, turns, hand size, penalties)
