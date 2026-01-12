# Codebase Structure

> Auto-generated developer documentation. Last updated: 2026-01-12

## Overview

Eleusis LLM Benchmark evaluates language models on **pattern discovery** using a card game. An LLM player receives cards and must deduce a hidden rule (e.g., "alternating colors") by observing which cards are accepted or rejected. The benchmark measures how efficiently models can form and test hypotheses.

**Architecture**: Game engine (pure Python) + LLM providers (OpenRouter, HuggingFace) + evaluation scripts.

## Tech Stack

- **Python 3.11+** with type hints
- **uv** for package management
- **pydantic** for data validation
- **openai** SDK for API calls (used by both providers)
- **PyYAML** for configuration
- **pandas/matplotlib** for analysis (optional)

## Directory Structure

```
src/eleusis/          # Main package
  game/               # Game engine (cards, state, rules)
  llm/                # LLM providers and player logic
  prompts/            # Prompt templates
  runner.py           # Round orchestration
  utils.py            # Logging, utilities

scripts/              # CLI entry points
  evaluate_single.py  # Main evaluation script
  analyze_results.py  # Post-hoc analysis
  generate_rule_library.py  # Rule generation
  evaluate_rules.py   # Rule statistics

tests/                # pytest tests
results/              # Evaluation outputs (JSON)
logs/                 # Debug logs
```

## Entry Points

### Primary: `scripts/evaluate_single.py`

Main evaluation script that runs multiple rounds of the game.

```bash
uv run scripts/evaluate_single.py --player "openrouter:anthropic/claude-haiku" --num-rounds 10
```

**Key functions**:
- `main()` - Orchestrates full evaluation, handles checkpointing
- `load_checkpoint()` / `validate_resume_config()` - Resume support
- `play_round()` - Delegates to `eleusis.runner.play_round()`

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_rule_library.py` | Generate rules.json with LLM |
| `scripts/evaluate_rules.py` | Calculate acceptance rates for rules |
| `scripts/analyze_results.py` | Cross-model analysis and plots |

## Core Modules

### `game/` - Game Engine

Pure Python game logic with no LLM dependencies.

#### `game/cards.py` - Card System
- **Classes**: `Suit` (enum), `Card` (frozen dataclass), `Deck`, `Hand`
- **Key details**:
  - Ranks 1-13 (Ace=1, King=13)
  - Double deck (104 cards)
  - `Card.color` property returns "red" or "black"
  - `Deck.shuffle(seed=N)` for reproducibility

#### `game/state.py` - Game State
- **Classes**: `GameState`, `PlayerState`, `Mainline`, `Sideline`
- **Key pattern**: Single-player focused
  ```python
  state = GameState("Player1")
  state.player.hand  # Access hand
  state.mainline.get_all()  # Accepted cards
  state.sidelines  # Dict of rejected cards by position
  ```

#### `game/engine.py` - Core Engine
- **Classes**: `Rule`, `GameEngine`, `PlayCardAction`, `GuessRuleAction`
- **Key methods**:
  - `Rule.evaluate(card, mainline) -> bool` - Execute rule code
  - `GameEngine.setup_game(seed)` - Deal hands, place starter
  - `GameEngine.play_turn(action) -> dict` - Process action
  - `GameEngine.calculate_score(max_turns, current_turn) -> int`

**Rule compilation** (line 39-88):
```python
# Rules are function bodies only, wrapped into:
def evaluate_rule(card, mainline):
    {code}

# Safe globals limit available builtins
safe_globals = {"len": len, "sum": sum, "any": any, ...}
```

#### `game/validator.py` - Rule Validation
- **Classes**: `RuleValidator`, `RuleFactory`, `ValidationResult`
- **Key methods**:
  - `RuleValidator.compare_rules()` - Simulation-based equivalence
  - `RuleFactory.create_rule_with_metadata()` - Load from library

**Equivalence checking** (line 135-214): Simulates gameplay with both rules, checking if they agree on all 52 cards across multiple turns.

### `llm/` - LLM Integration

#### `llm/base.py` - Base Client
- **Classes**: `BaseLLMClient` (ABC), `LLMCallMetrics`, `GenerateMetrics`
- **Key methods**:
  - `generate(prompt, xml_tag, return_dict)` - Main generation with auto-continuation
  - `convert_rule_to_code(rule_text)` - Natural language to Python
  - `get_usage_stats()` - Token counts, costs

**Auto-continuation** (line 159-256): If response is truncated (`finish_reason="length"`), automatically prompts for continuation with escalating strategies.

#### `llm/player.py` - LLMScientist
- **Class**: `LLMScientist`
- **Key methods**:
  - `get_action(game_state) -> Action` - Decide what to play
  - `_select_move()` - LLM-based card selection
  - `record_action_result()` - Track play history

**Card parsing** (line 108-152): Converts LLM output like "5♥" to `Card(5, Suit.HEARTS)`.

#### `llm/openrouter.py`, `llm/huggingface.py` - Providers
Implement `_call_api()` for respective APIs. Both use openai-compatible SDK.

#### `llm/__init__.py` - Factory
```python
def create_client(model_spec: str, ...) -> BaseLLMClient:
    # "openrouter:model" -> OpenRouterClient
    # "hf:model" -> HuggingFaceClient
```

### `prompts/` - Prompt Templates

#### `prompts/action.py`
- `get_action_prompt()` - Player turn prompt with game state, hand, history

#### `prompts/game_rules.py`
- `get_game_rules()` - Full game explanation
- `get_eleusis_rules()` - Card game basics

#### `prompts/compile.py`
- `get_rule_compile_prompt()` - Convert rule description to Python
- `get_library_generation_prompt()` - Generate multiple rules

### `runner.py` - Round Orchestration

**Function**: `play_round(config, round_number, rule, ...) -> dict`

Orchestrates a single round:
1. Initialize LLM clients (rule compiler + player)
2. Load/create rule via `RuleFactory`
3. Setup game state via `GameEngine.setup_game()`
4. Main loop: `LLMScientist.get_action()` → `GameEngine.play_turn()`
5. Return results with turn data, LLM usage, timing

## Execution Flow

```
evaluate_single.py::main()
    ↓
    load config.yaml
    ↓
    for each round:
        ↓
        runner.play_round()
            ↓
            create_client() × 2 (rule_compiler, player)
            ↓
            RuleFactory.create_rule_with_metadata()
            ↓
            GameEngine.setup_game()
            ↓
            while not game_over:
                LLMScientist.get_action()
                    ↓
                    get_action_prompt() → LLM → parse card
                ↓
                GameEngine.play_turn()
                    ↓
                    If PlayCardAction: evaluate, update mainline/sideline
                    If GuessRuleAction: RuleValidator.compare_rules()
            ↓
            return round results
        ↓
        save checkpoint to results/
```

## Data Flow

```
rules.json (pre-generated)
    ↓ RuleFactory
Rule object
    ↓ GameEngine.setup_game()
GameState (mainline, hand, deck)
    ↓ LLMScientist.get_action()
Prompt → LLM API → Action
    ↓ GameEngine.play_turn()
Updated GameState + Result dict
    ↓
results/{folder}/results.json
```

## Configuration

### `config.yaml` Structure

```yaml
providers:           # API key env vars
game:               # num_rounds, max_turns, hand_size, wrong_guess_penalty
rule_compiler:      # Model for rule compilation
rules:              # library_path, selection (sequential/random), index
llm:                # max_tokens, temperature, retries
model:              # Player model spec (e.g., "openrouter:claude-haiku")
seed:               # For reproducibility
```

### Environment Variables

```bash
OPENROUTER_API_KEY=...
HF_TOKEN=...
```

## Testing

```bash
uv run pytest                          # All tests
uv run pytest tests/test_game_engine.py  # Specific file
uv run pytest -k "test_play_card"      # Pattern match
```

**Test files**:
- `test_game_engine.py` - GameEngine, scoring, guessing
- `test_python_rule.py` - Rule compilation, sandbox security

## Build & Development

```bash
uv sync                    # Install dependencies
uv run ruff check src/     # Lint
uv run pytest              # Test
```

## Key Patterns & Conventions

### Model Specification
```
"openrouter:anthropic/claude-haiku"  → OpenRouter
"hf:meta-llama/Llama-3.3-70B"        → HuggingFace
```

### Rule Format
Rules are **function bodies only** (no `def`):
```python
# Correct
if not mainline:
    return True
return card.color != mainline[-1].color

# Wrong - don't include def
def evaluate_rule(card, mainline):
    ...
```

### Available in Rules
- `card.rank` (1-13), `card.color` ("red"/"black"), `card.suit.suit_name`
- `mainline` (list of previous accepted cards)
- Builtins: `len`, `sum`, `min`, `max`, `any`, `all`, `range`, `set`, etc.

### Structured Output
LLM responses use XML tags:
```xml
<ACTION>{"card": "5♥", "reasoning_summary": "...", ...}</ACTION>
```

### Checkpointing
Results saved after each round to `results/{folder}/results.json` with `checkpoint` field for resume.

## Common Tasks

### Add a New LLM Provider

1. Create `src/eleusis/llm/newprovider.py`
2. Inherit from `BaseLLMClient`
3. Implement `_call_api()` and `provider_name`
4. Register in `llm/__init__.py::create_client()`

### Add a New Prompt

1. Create function in appropriate `prompts/*.py` file
2. Export in `prompts/__init__.py`
3. Call from `llm/player.py` or `llm/base.py`

### Modify Scoring

Edit `GameEngine.calculate_score()` in `game/engine.py:252-269`.

### Add Analysis Metric

Edit `scripts/analyze_results.py`, add function, call from `main()`.
