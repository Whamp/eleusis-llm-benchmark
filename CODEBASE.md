# Codebase Structure

> Auto-generated developer documentation. Last updated: 2026-01-16

## Overview

Eleusis LLM Benchmark evaluates language models on **pattern discovery** using a card game. An LLM player receives cards and must deduce a hidden rule (e.g., "alternating colors") by observing which cards are accepted or rejected. The benchmark measures how efficiently models can form and test hypotheses.

**Architecture**: Game engine (pure Python) + LLM providers (Anthropic, OpenAI, Google, xAI, HuggingFace) + evaluation scripts.

## Tech Stack

- **Python 3.11+** with type hints
- **uv** for package management
- **pydantic** for data validation
- **anthropic**, **openai**, **google-genai** SDKs for API calls
- **huggingface-hub** for open model inference
- **PyYAML** for configuration
- **pandas/matplotlib** for analysis (optional)

## Directory Structure

```
src/eleusis/          # Main package
  game/               # Game engine (cards, state, rules)
  llm/                # LLM providers and player logic
  prompts/            # Prompt templates
  runner.py           # Round orchestration
  player.py           # LLM-based player
  utils.py            # Logging, utilities

scripts/              # CLI entry points
  evaluate_single.py  # Main evaluation script
  analyze_results.py  # Post-hoc analysis
  generate_rule_library.py  # Rule generation

tests/                # pytest tests
results/              # Evaluation outputs (JSON)
logs/                 # Debug logs

models.yaml           # Model configurations (provider, model_id, options)
config.yaml           # Game settings (rounds, turns, rules)
rules.json            # Pre-generated rule library
```

## Entry Points

### Primary: `scripts/evaluate_single.py`

Main evaluation script that runs multiple rounds of the game.

```bash
uv run scripts/evaluate_single.py --model "claude-opus-4.5" --num-rounds 10
```

**Key functions**:
- `main()` - Orchestrates full evaluation, handles checkpointing
- `load_checkpoint()` / `reconstruct_config_from_checkpoint()` - Resume support (self-contained, no config.yaml needed)
- `preflight_check()` - Fast-fail connectivity test before evaluation
- Delegates to `eleusis.runner.play_round()`

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_rule_library.py` | Compile human-written rules from rules.txt to rules.json |
| `scripts/analyze_results.py` | Cross-model analysis and plots |

## Core Modules

### `game/` - Game Engine

Pure Python game logic with no LLM dependencies.

#### `game/cards.py` - Card System
- **Location**: `src/eleusis/game/cards.py`
- **Classes**: `Suit` (enum), `Card` (frozen dataclass), `Deck`, `Hand`
- **Key details**:
  - Ranks 1-13 (Ace=1, King=13)
  - Double deck (104 cards)
  - `Card.color` property returns "red" or "black"
  - `Deck.shuffle(seed=N)` for reproducibility

#### `game/state.py` - Game State
- **Location**: `src/eleusis/game/state.py`
- **Classes**: `GameState`, `PlayerState`, `Mainline`, `Sideline`
- **Key pattern**: Single-player focused
  ```python
  state = GameState("Player1")
  state.player.hand  # Access hand
  state.mainline.get_all()  # Accepted cards
  state.sidelines  # Dict of rejected cards by position
  state.to_compact_string()  # "5♥ [3♠] 7♦" format
  ```

#### `game/engine.py` - Core Engine
- **Location**: `src/eleusis/game/engine.py`
- **Classes**: `Rule`, `GameEngine`, `PlayCardAction`, `GuessRuleAction`
- **Key methods**:
  - `Rule.evaluate(card, mainline) -> bool` - Execute rule code in sandbox
  - `GameEngine.setup_game(round_seed)` - Deal hands, place starter
  - `GameEngine.play_turn(action) -> dict` - Process action
  - `GameEngine.calculate_score(max_turns, current_turn) -> int`

**Rule compilation** (lines 39-96):
```python
# Rules are function bodies only, wrapped into:
def evaluate_rule(card, mainline):
    {code}

# Safe globals limit available builtins (no imports allowed)
safe_globals = {"len": len, "sum": sum, "any": any, "all": all, ...}
```

#### `game/validator.py` - Rule Validation
- **Location**: `src/eleusis/game/validator.py`
- **Classes**: `RuleValidator`, `RuleFactory`, `ValidationResult`
- **Key methods**:
  - `RuleValidator.compare_rules()` - Simulation-based equivalence testing
  - `RuleValidator.check_equivalence_by_simulation()` - Tests all 52 cards across multiple simulated turns
  - `RuleFactory.create_rule_with_metadata()` - Load from library (sequential or random)

**Equivalence checking** (lines 141-220): Simulates gameplay with both rules, checking if they agree on all 52 cards across multiple turns. Returns mismatch details for debugging.

#### `game/metrics.py` - Rule Metrics
- **Location**: `src/eleusis/game/metrics.py`
- **Functions**: `code_complexity()` - AST node count + cyclomatic complexity
- **Classes**: `RuleEvaluator` - Simulates random plays to calculate acceptance rates

### `llm/` - LLM Integration

#### `llm/base.py` - Base Client
- **Location**: `src/eleusis/llm/base.py`
- **Classes**: `BaseLLMClient` (ABC), `LLMCallMetrics`, `GenerateMetrics`, `TruncationError`
- **Key methods**:
  - `generate(prompt, xml_tag, return_dict)` - Main generation, raises `TruncationError` on max tokens
  - `convert_rule_to_code(rule_text)` - Natural language to Python
  - `get_usage_stats()` - Token counts (prompt, output, reasoning, answer)
  - `_extract_content_from_response()` - XML tag and code block extraction

**Truncation handling**: If `finish_reason="length"`, raises `TruncationError`. Retry logic is handled at the player level.

#### `llm/__init__.py` - Client Factory
- **Location**: `src/eleusis/llm/__init__.py`
- **Function**: `create_client(model_key, temperature, max_tokens, role, seed)`
- **Function**: `load_model_config(model_key)` - Load from models.yaml

Routes model keys to appropriate providers:
```python
create_client("claude-opus-4.5")    # → AnthropicClient
create_client("gpt-5.2-medium")     # → OpenAIClient
create_client("gemini-3-pro")       # → GoogleClient
create_client("grok-4")             # → XAIClient
create_client("deepseek-r1")        # → HuggingFaceClient
```

#### Provider Implementations

| File | Provider | Key Config |
|------|----------|------------|
| `llm/anthropic.py` | Anthropic | `reasoning_budget` for extended thinking |
| `llm/openai_client.py` | OpenAI | `reasoning_effort` (none→xhigh) |
| `llm/google.py` | Google | `thinking_level` (low/high) |
| `llm/xai.py` | xAI | Standard completion |
| `llm/huggingface.py` | HuggingFace | `hf_provider`, `reasoning_format` |

All implement `_call_api()` returning `(response, LLMCallMetrics)`.

#### `player.py` - LLMScientist
- **Location**: `src/eleusis/player.py`
- **Class**: `LLMScientist`
- **Key methods**:
  - `get_action(game_state) -> Action` - Decide what to play
  - `_select_move()` - LLM-based card selection with retry logic
  - `_parse_card()` - Convert "5♥" to Card object
  - `record_action_result()` - Track play history

**Retry logic**: Up to 3 attempts per turn. On retries, appends "DO NOT REASON TOO LONG" hint to prompt. Tracks causes: `max_token_reached`, `card_parse_error`, `other_error`. Falls back to random card after 3 failures.

### `prompts/` - Prompt Templates

#### `prompts/action.py`
- **Function**: `get_action_prompt()` - Player turn prompt with game state, hand, history
- Returns structured prompt expecting `<ACTION>{"card": "5♥", ...}</ACTION>` response

#### `prompts/compile.py`
- **Function**: `get_rule_compile_prompt()` - Convert rule description to Python code
- Includes sandbox restrictions and examples

#### `prompts/game_rules.py`
- **Function**: `get_game_rules()` - Full game explanation for LLM context

### `runner.py` - Round Orchestration

- **Location**: `src/eleusis/runner.py`
- **Function**: `play_round(config, round_number, rule, ...) -> dict`

Orchestrates a single round:
1. Initialize LLM clients (rule compiler + player)
2. Load/create rule via `RuleFactory`
3. Setup game state via `GameEngine.setup_game()`
4. Main loop: `LLMScientist.get_action()` → `GameEngine.play_turn()`
5. Return results with turn data, LLM usage, timing

### `utils.py` - Utilities

- **Location**: `src/eleusis/utils.py`
- **Functions**:
  - `model_spec_to_display_name()` - "claude-opus-4.5" → "Claude Opus 4.5"
  - `setup_logging()` - Dual output: colored console + detailed file
- **Classes**: `ColoredFormatter` - ANSI color codes for terminal

## Execution Flow

```
evaluate_single.py::main()
    ↓
    load config.yaml + models.yaml
    ↓
    preflight_check(model_key)  # Fast-fail connectivity test
    ↓
    for each round:
        ↓
        runner.play_round()
            ↓
            create_client() × 2 (rule_compiler, player)
            ↓
            RuleFactory.create_rule_with_metadata()
            ↓
            GameEngine.setup_game(round_seed)
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
models.yaml (provider configs)
    ↓ create_client()
LLM clients (rule_compiler, player)

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

### `models.yaml` Structure

Define model configurations with provider-specific settings:

```yaml
# Closed providers
claude-opus-4.5:
  provider: anthropic
  model_id: claude-opus-4-5-20251101
  reasoning_budget: 8000

gpt-5.2-medium:
  provider: openai
  model_id: gpt-5.2
  reasoning_effort: medium

# Open models via HuggingFace
deepseek-r1:
  provider: huggingface
  model_id: deepseek-ai/DeepSeek-R1
  hf_provider: together
  reasoning_format: think_tags
```

### `config.yaml` Structure

```yaml
game:               # num_rounds, max_turns, hand_size, wrong_guess_penalty, seed
rule_compiler:      # Model key for rule compilation
rules:              # library_path, selection (sequential/random), index
llm:                # max_tokens, temperature, max_llm_retries, seed
```

### Environment Variables

```bash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
XAI_API_KEY=...
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

Use model keys from `models.yaml`:
```bash
--model "claude-opus-4.5"    # Anthropic
--model "gpt-5.2-high"       # OpenAI with high reasoning
--model "deepseek-r1"        # HuggingFace via Together
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
- No imports allowed

### Structured Output

LLM responses use XML tags:
```xml
<ACTION>{"card": "5♥", "reasoning_summary": "...", "tentative_rule": "...", "confidence_level": 7, "guess_rule": false}</ACTION>
```

### Checkpointing

Results saved after each round to `results/{folder}/results.json` with `checkpoint` field for resume. Checkpoints are self-contained (include full rules library).

### Results JSON Structure

Each turn in `results.json` includes:
```json
{
  "turn_number": 1,
  "llm_response": {"card": "5♥", "reasoning_summary": "..."},
  "action_result": {"card": "5♥", "accepted": true},
  "tokens": {"output_tokens": 500, "reasoning_tokens": 400, "answer_tokens": 100},
  "retry_count": 0,
  "retry_causes": [],
  "guess_attempt": null
}
```

**Statistics include**:
- `total_retries`: Sum of all retry counts
- `retry_by_cause`: Breakdown by cause type
- `total_output_tokens`, `total_reasoning_tokens`, `total_answer_tokens`

## Common Tasks

### Add a New LLM Provider

1. Create `src/eleusis/llm/newprovider.py`
2. Inherit from `BaseLLMClient`
3. Implement `_call_api()` and `provider_name` property
4. Register in `llm/__init__.py::create_client()`

### Add a New Model

1. Add entry to `models.yaml` with provider and model_id
2. Include any provider-specific options (reasoning_budget, etc.)

### Modify Scoring

Edit `GameEngine.calculate_score()` in `game/engine.py:268-285`.

### Add Analysis Metric

Edit `scripts/analyze_results.py`, add function, call from `main()`.
