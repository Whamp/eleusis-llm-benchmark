# Eleusis LLM Benchmark

A benchmark for evaluating LLMs on pattern discovery in card games. Models attempt to deduce secret rules by testing cards against hidden patterns.

## Quick Start

```bash
# Install dependencies
uv sync

# Set up API keys in .env file (only the ones you need)
echo "ANTHROPIC_API_KEY=your_key" > .env
echo "OPENAI_API_KEY=your_key" >> .env
echo "GOOGLE_API_KEY=your_key" >> .env
echo "XAI_API_KEY=your_key" >> .env
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

# Override player model (using model key from models.yaml)
uv run scripts/evaluate_single.py --player "claude-opus"

# Test 20 rules with a specific model and custom tag
uv run scripts/evaluate_single.py --player "gpt-5.2" --num-rules 20 --tag gpt

# Start from specific rule index
uv run scripts/evaluate_single.py --rule-index 10

# Resume interrupted evaluation
uv run scripts/evaluate_single.py --resume results/solo_evaluation_20251205_151306
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--config FILE` | Config file path (default: config.yaml) |
| `--player MODEL` | Model key from models.yaml (e.g., `claude-opus`, `gpt-5.2`) |
| `--num-rules N` | Number of distinct rules to test |
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
# Use model keys from models.yaml
claude-opus
gpt-5.2
deepseek-r1
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

### models.yaml

Model configurations with provider-specific settings:

```yaml
# Closed providers - direct API access
claude-opus:
  provider: anthropic
  model_id: claude-opus-4-5-20251101
  reasoning_budget: 8192  # Extended thinking budget

gpt-5.2:
  provider: openai
  model_id: gpt-5.2
  reasoning_effort: medium  # none|minimal|low|medium|high|xhigh

gemini-3-pro:
  provider: google
  model_id: gemini-3-pro
  thinking_level: high  # low|high

grok-4:
  provider: xai
  model_id: grok-4-fast-reasoning

# Open models - HuggingFace Inference Providers
deepseek-r1:
  provider: huggingface
  model_id: deepseek-ai/DeepSeek-R1
  hf_provider: together  # or novita, etc.
  reasoning_format: think_tags  # think_tags or separate_field
```

### config.yaml

```yaml
game:
  num_rules: 50         # Number of distinct rules to test (0 = use entire library)
  num_rounds_per_rule: 1  # How many rounds to play with each rule
  max_turns: 40
  hand_size: 12
  wrong_guess_penalty: 2

rule_compiler:
  model: gpt-oss-120b  # References key in models.yaml

rules:
  library_path: "rules.json"
  selection: "sequential"

llm:
  max_tokens: 16384
  temperature: 0.7

model: deepseek-r1  # Model key from models.yaml
```

### Environment Variables

Create a `.env` file with the API keys you need:

```
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
XAI_API_KEY=your_xai_key
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

## Data Analysis

Analyze evaluation results across multiple models:

```bash
uv run scripts/analyze_results.py
```

Outputs are saved to `results/analysis/`.

### Analyses Performed

**1. Basic Model Comparison**
Compare models on success rate, average score, turns to completion, and token efficiency (score per 1K tokens). Identifies which models solve rules most efficiently.

**2. Confidence Calibration**
Measure how well models' self-reported confidence (0-10) predicts actual guess accuracy. Detects overconfidence by comparing mean confidence when correct vs wrong.

**3. Rule Complexity Analysis**
Correlate success rate with rule complexity using two metrics from `rules.json`:
- **Cyclomatic complexity**: Number of decision branches in the rule code
- **Node count**: AST size (total nodes in the parsed code)

Also analyzes success rate vs rule selectivity (acceptance rate).

**4. Learning Curves**
Track confidence trajectory and card acceptance rate over turns to understand how models learn during a round.

### Output Files

| File | Description |
|------|-------------|
| `basic_metrics.csv` | Model comparison table |
| `basic_metrics.png` | Bar charts for key metrics |
| `confidence_calibration.png` | Calibration curve + confidence distributions |
| `complexity_analysis.png` | Success vs complexity (cyclomatic + node count) |
| `learning_curves.png` | Metrics over turn progression |

## Architecture

### Core Components (src/eleusis/)

| File | Description |
|------|-------------|
| `cards.py` | Card representation (rank 1-13, 4 suits) |
| `game_state.py` | Game state: mainline, sidelines, hands, deck |
| `game_engine.py` | Game engine with Rule class |
| `runner.py` | Solo round orchestration |
| `llm/` | LLM clients (Anthropic, OpenAI, Google, xAI, HuggingFace) |
| `prompts/` | Prompt templates |

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
- LLM retries: Up to 3 attempts per turn with fallback to random card; retry causes tracked in results.json (`max_token_reached`, `card_parse_error`, `other_error`)
