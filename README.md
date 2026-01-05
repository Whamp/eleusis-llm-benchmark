# Eleusis LLM Benchmark

A benchmark for evaluating LLMs on pattern discovery in card games. Models attempt to deduce secret rules by testing cards against hidden patterns.

## Quick Start

```bash
# Install dependencies
uv sync

# Set up API keys in .env file
echo "OPENROUTER_API_KEY=your_key" > .env
echo "HF_TOKEN=your_hf_token" >> .env

# Run solo evaluation (single model)
uv run scripts/evaluate_single.py

# Run parallel evaluation (multiple models)
./scripts/run_parallel_eval.sh
```

## Solo Mode (Primary)

A single LLM player attempts to discover a hidden pattern as efficiently as possible.

### Running Evaluations

```bash
# Default: uses config.yaml settings
uv run scripts/evaluate_single.py

# Override player model
uv run scripts/evaluate_single.py --player "openrouter:anthropic/claude-3.5-haiku"

# Custom rounds and tag for identification
uv run scripts/evaluate_single.py --player "openrouter:google/gemini-2.0-flash-001" \
    --num-rounds 20 --tag gemini

# Start from specific rule index
uv run scripts/evaluate_single.py --rule-index 10

# Resume interrupted evaluation
uv run scripts/evaluate_single.py --resume results/solo_evaluation_20251205_151306
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--config FILE` | Config file path (default: config.yaml) |
| `--player MODEL` | Model spec (e.g., `openrouter:anthropic/claude-haiku`) |
| `--player-name NAME` | Display name (auto-generated if not provided) |
| `--num-rounds N` | Number of rounds to play |
| `--rule-index N` | Starting rule index (for sequential selection) |
| `--max-turns N` | Max turns per round |
| `--tag TAG` | Tag for output folder identification |
| `--resume PATH` | Resume from checkpoint folder |

### Parallel Evaluation

Run multiple models in parallel:

```bash
# Run default models (edit DEFAULT_MODELS in script)
./scripts/run_parallel_eval.sh

# Use custom models file
./scripts/run_parallel_eval.sh -m models.txt

# 10 rounds each, max 2 parallel jobs
./scripts/run_parallel_eval.sh -n 10 -j 2
```

**Options:**
- `-m, --models FILE` - File with model specs (one per line)
- `-n, --num-rounds N` - Rounds per evaluation
- `-r, --rule-index N` - Starting rule index
- `-j, --jobs N` - Max parallel jobs
- `-c, --config FILE` - Base config file

**Models file format** (see `models.txt.example`):
```
# Comments start with #
openrouter:anthropic/claude-3.5-haiku
openrouter:google/gemini-2.0-flash-001
openrouter:openai/gpt-4o-mini
```

### Game Mechanics

- Constant hand size (12 cards) - draw 1 after each play
- Play a card each turn (no "no play" action)
- Guess the rule at any time
- Game ends on correct guess or max turns (40)

### Scoring

- Correct guess: `score = max_turns - current_turn - (3 × failed_guesses)`
- No correct guess: `score = 0`
- Higher is better (rewards efficiency)

### Resume Feature

Evaluations checkpoint after each round and can be resumed:

```bash
uv run scripts/evaluate_single.py --resume results/solo_evaluation_20251205_151306
```

Requirements:
- Only works with `selection: "sequential"` in config
- Config parameters must match original run
- Checkpoints are self-contained (include all rules)

## Configuration

### config.yaml

```yaml
providers:
  openrouter:
    api_key_env: "OPENROUTER_API_KEY"
  huggingface:
    api_key_env: "HF_TOKEN"

models:
  game_master:
    name: "hf:openai/gpt-oss-120b"  # For rule compilation

rule_source:
  library_path: "rules.json"
  selection: "sequential"  # or "random"
  index: 0                 # starting index
  min_acceptance: 0.15     # filter rules by acceptance rate
  max_acceptance: 0.55

solo_game:
  num_rounds: 50
  max_turns: 40
  hand_size: 12
  wrong_guess_penalty: 2
  player:
    name: "openrouter:anthropic/claude-3.5-haiku"
    temperature: 0.7
```

### Model Specification

Models are specified with provider prefix:
- `openrouter:anthropic/claude-3.5-haiku` - OpenRouter API
- `hf:meta-llama/Llama-3.3-70B` - HuggingFace Inference Providers

### Environment Variables

Create a `.env` file:
```
OPENROUTER_API_KEY=your_openrouter_key
HF_TOKEN=your_huggingface_token
```

## Rule Library

Rules are loaded from a pre-generated JSON library:

```bash
# Generate new rules
uv run scripts/generate_rule_library.py --num-rules 50 --output rules.json

# Evaluate rule acceptance rates
uv run scripts/evaluate_rules.py --library rules.json
```

Rules are filtered by acceptance rate (configurable in config.yaml) to ensure playable difficulty.

## Architecture

### Core Components (src/eleusis/)

| File | Description |
|------|-------------|
| `cards.py` | Card representation (rank 1-13, 4 suits) |
| `game_state.py` | Game state: mainline, sidelines, hands, deck |
| `game_engine.py` | Game engine with Rule class |
| `game_runner_solo.py` | Solo round orchestration |
| `providers/` | LLM clients (OpenRouter, HuggingFace) |
| `prompts.py` | Prompt templates |

### Key Concepts

- **Mainline**: Row of accepted cards (visible to player)
- **Sideline**: Rejected cards shown below mainline
- **Rule**: Deterministic Python function evaluating `(card, mainline) → bool`

### Rule Compilation

Rules are compiled into sandboxed Python with limited builtins:
- Allowed: `len`, `sum`, `min`, `max`, `abs`, `any`, `all`
- Available: `card.rank`, `card.color`, `card.suit.suit_name`, `mainline`

---

## Development

- Rules must be function body only (no `def`)
- Rule comparison uses simulation-based equivalence testing
- Failed guesses are tracked to prevent duplicates
- Logging: Python logging with console/file levels
