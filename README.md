# Eleusis LLM Benchmark

A benchmark for evaluating Large Language Models on **inductive reasoning and pattern discovery**, using an adaptation of Robert Abbott's card game [Eleusis](https://en.wikipedia.org/wiki/Eleusis_(card_game)) (1956).

A secret rule determines which cards are accepted. The model must discover the rule by playing cards and observing the outcomes, mimicking the process of scientific hypothesis testing.

Full results on the dedicated [Eleusis space on Hugging Face](https://huggingface.co/spaces/huggingface/eleusis-benchmark)

<p align="center">
  <img src="results.png" alt="Eleusis benchmark results" style="width: 70%; max-width: 900px;">
</p>


## Quick Start

```bash
# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Set up API keys (only the providers you need)
cp .env.example .env
# Edit .env and set at least HF_TOKEN

# Pick the same model you plan to use for the full benchmark
MODEL="kimi-k2"

# Run a smoke test to verify auth + config end-to-end for that model
uv run python scripts/evaluate_single.py \
  --config config.smoke.yaml \
  --model "$MODEL" \
  --tag smoke

# If the smoke test succeeds, run the full benchmark with the same model
uv run python scripts/evaluate_single.py --model "$MODEL"
```

The smoke test runs one short round on a simple known rule. It is only meant to verify
that your Hugging Face token, rule compiler, and chosen benchmark model are wired
correctly before starting a much longer run. It is not a quality benchmark.

## Two Model Roles

This benchmark uses two separate model configurations:

1. **Rule compiler** — converts natural-language rules into Python code for execution.
   By default this is configured in `config.yaml` as `openai/gpt-oss-120b` via
   Hugging Face Inference Providers.
2. **Benchmark model** — the model actually playing Eleusis. This is selected with
   `--model <key>` and loaded from `models.yaml`.

Example:

```bash
uv run python scripts/evaluate_single.py --model "kimi-k2"
```

In that command, `kimi-k2` is the benchmark model. The rule compiler still comes from
`config.yaml` unless you change it.

## How It Works

Each evaluation round plays out as follows:

1. A **secret rule** is loaded (a Python function that accepts or rejects cards based on their properties and the sequence of previously played cards)
2. The model receives a hand of 12 cards and sees a starter card on the table
3. Each turn, the model **plays a card** and observes if it's accepted (mainline) or rejected (sideline)
4. The model can **guess the rule** at any point, at the cost of a penalty for wrong guesses
5. The round ends when the model guesses correctly or reaches the turn limit

**Scoring:** `max_turns - turn_used - (penalty × wrong_guesses)` for a correct guess. Higher is better, score is floored at 0.

## Running Evaluations

### Smoke Test

Use the smoke test before a full run, with the same model you intend to benchmark:

```bash
MODEL="kimi-k2"
uv run python scripts/evaluate_single.py \
  --config config.smoke.yaml \
  --model "$MODEL" \
  --tag smoke
```

This uses:
- your chosen benchmark model from `--model`
- the default rule compiler from `config.smoke.yaml`: `openai/gpt-oss-120b`
- one easy rule from `rules.json`: `even_ranks_only`

A successful smoke test means the command completes and writes a results folder under
`results/` for the same model you plan to use in the full benchmark. The benchmark
model does not need to solve the rule perfectly.

### Single Model


| Argument | Description |
|----------|-------------|
| `--model MODEL` | Model key from `models.yaml` (required unless `--resume`) |
| `--num-rules N` | Number of distinct rules to test (default: all) |
| `--rule-index N` | Starting rule index |
| `--max-turns N` | Max turns per round (default: 30) |
| `--tag TAG` | Tag appended to output folder name |
| `--resume PATH` | Resume from checkpoint folder |
| `--config FILE` | Config file path (default: `config.yaml`) |
| `--batch-round-offset N` | Run 1 round per rule with batch index N (for parallel workers) |

### Parallel Workers (Single Model)

A full benchmark with 26 rules × 3 rounds can take tens of hours. Split the work
across parallel workers, each playing every rule once with a different deck shuffle:

```bash
uv run python scripts/run_parallel_benchmark.py \
  --model "kimi-k2" \
  --workers 3
```

This launches 3 processes. Each worker plays all 26 rules exactly once (26 rounds
per worker). The workers stay balanced because they all play the same rules — when
a hard rule slows one worker down, it slows all of them equally.

Under the hood, each worker runs `evaluate_single.py` with a different
`--batch-round-offset` (0, 1, or 2). The offset feeds into the deck seed so the
same rule produces a different shuffle per worker.

Preview the commands without running:

```bash
uv run python scripts/run_parallel_benchmark.py --model "kimi-k2" --workers 3 --dry-run
```

Results land in separate folders (`results/solo_evaluation_*_w0_*`,
`results/solo_evaluation_*_w1_*`, etc.) and are merged automatically by
`analyze_results.py`.

### Pause and Resume

New runs use `benchmark_run.sqlite3` as their authoritative store and regenerate
`results.json` after each completed Round. Setup is committed before the first
Model Attempt, and each validated Turn is committed before the next Model
Attempt. Resume restores the last committed board, hand, deck, RNG state, and
seed, then starts the exact next Turn without repeating completed work.

**Pause:** Kill the processes (`Ctrl+C`, or `kill <pid>`).

**Resume:** Point `--resume` at each worker's result folder. When SQLite is
present, resume uses it instead of `results.json` and verifies the stored model,
compiler, prompts, settings, schedule, seeds, and source fingerprint. A changed
scientific input requires a new Benchmark Run. Historical JSON-only folders keep
the legacy completed-Round resume path.

```bash
# Find the result folders from the interrupted run
ls results/solo_evaluation_*_w*_kimi-k2/

# Resume each worker
uv run python scripts/evaluate_single.py \
  --resume results/solo_evaluation_20260403_120000_w0_kimi-k2 &
uv run python scripts/evaluate_single.py \
  --resume results/solo_evaluation_20260403_120000_w1_kimi-k2 &
uv run python scripts/evaluate_single.py \
  --resume results/solo_evaluation_20260403_120000_w2_kimi-k2 &
```

The stored model and seed inputs are reused. If the Run used a non-default
configuration path, pass the same `--config` so resume can verify it. Do not
re-specify scientific overrides such as `--model` or `--batch-round-offset` with
different values.

### Multiple Models in Parallel

```bash
# Provide a file listing model keys (one per line)
./scripts/run_parallel_eval.sh eval_models.txt

# With a custom config
./scripts/run_parallel_eval.sh eval_models.txt custom_config.yaml
```

The models file contains one model key per line (lines starting with `#` are ignored).

### Analyzing Results

```bash
uv run python scripts/analyze_results.py results/<folder>
```

Generates charts and tables: basic metrics comparison, complexity analysis, per-model reports, token usage, and more.

### Status Reports (In-Progress Benchmarks)

Compare a running benchmark against completed reference models. Merges worker results,
identifies completed rules (rules with all rounds across workers), filters all
models to those rules, and runs the full analysis pipeline.

```bash
uv run python scripts/status_report.py \
  --reference results/260312_all_models_corrected \
  results/solo_evaluation_*_w*_rys-qwen3.5-27b-fp8-xl
```

Output goes to `status/` inside the first worker folder. Re-run anytime to get
an updated report with newly completed rules.

## Configuration

### `models.yaml` — Model Definitions

Each entry defines a model with its provider and provider-specific settings:

```yaml
# Closed-source providers
claude-opus-4.5:
  provider: anthropic
  model_id: claude-opus-4-5-20251101
  reasoning_budget: 16000

gpt-5.2-medium:
  provider: openai
  model_id: gpt-5.2
  reasoning_effort: medium    # none|minimal|low|medium|high|xhigh

gemini-3-pro-preview-high:
  provider: google
  model_id: gemini-3-pro-preview
  thinking_level: high        # low|high

grok-4:
  provider: xai
  model_id: grok-4

# Open models via HuggingFace Inference Providers
deepseek-r1:
  provider: huggingface
  model_id: deepseek-ai/DeepSeek-R1
  hf_provider: together
  reasoning_format: think_tags  # think_tags|separate_field
```

Supported providers: `anthropic`, `openai`, `google`, `xai`, `huggingface`, `openai_compat`.

Self-hosted models (vLLM, SGLang, llama.cpp) use the `openai_compat` provider:

```yaml
my-local-model:
  provider: openai_compat
  model_id: my-org/My-Model
  base_url: http://localhost:8000/v1
  api_key: sk-no-key-required
  reasoning_format: reasoning_content  # reasoning_content|think_tags
  timeout: 600
```

### `config.yaml` — Game Settings

```yaml
game:
  num_rules: 0              # 0 = use entire rule library
  num_rounds_per_rule: 3
  max_turns: 30
  hand_size: 12
  wrong_guess_penalty: 2
  seed: 42

rule_compiler:
  provider: huggingface
  model_id: openai/gpt-oss-120b
  hf_provider: together
  reasoning_format: separate_field

rules:
  library_path: "rules.json"
  selection: "sequential"

llm:
  max_tokens: 16384
  max_llm_retries: 3
  temperature: 0.7
```

`config.yaml` controls the game settings and the rule compiler. The benchmark model is
still chosen separately with `--model` and loaded from `models.yaml`.

### Environment Variables

Create a `.env` file with the API keys for the providers you use.

For the default setup, `HF_TOKEN` is required because the rule compiler uses Hugging
Face Inference Providers.

```dotenv
HF_TOKEN=hf_...

# Optional: bill Inference Providers calls to a Hugging Face org instead of your
# personal account.
# HF_BILL_TO=your_org_name

ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
XAI_API_KEY=
```

## Rule Library

Rules are written in natural language in `rules.txt` and compiled to executable Python via an LLM:

```bash
uv run python scripts/generate_rule_library.py --input rules.txt --output rules.json
```

Each compiled rule is a Python function body with access to `card` (current card) and `mainline` (list of previously accepted cards). Card properties: `card.rank` (1–13), `card.color` (`"red"` or `"black"`), `card.suit.suit_name` (`"hearts"`, `"diamonds"`, `"clubs"`, `"spades"`).

The game uses a double deck (104 cards).

## Project Structure

```
src/eleusis/
  game/
    cards.py          Card, Deck, Hand (double 52-card deck)
    state.py          GameState, PlayerState, Mainline, Sideline
    engine.py         Rule, GameEngine, scoring
    validator.py      RuleValidator, RuleFactory, simulation-based equivalence
    metrics.py        Rule complexity (cyclomatic, AST node count)
  llm/
    base.py           BaseLLMClient interface
    anthropic.py      Anthropic (extended thinking)
    openai_client.py  OpenAI (reasoning effort)
    google.py         Google (thinking levels)
    xai.py            xAI
    huggingface.py    HuggingFace Inference Providers
  prompts/            Prompt templates (action, rule compilation, game rules)
  analysis/           Result analysis and visualization
  player.py           LLMScientist — main player logic
  runner.py           Round orchestration

scripts/
  evaluate_single.py         Single-model evaluation
  run_parallel_benchmark.py  Parallel multi-worker evaluation (single model)
  run_parallel_eval.sh       Parallel multi-model evaluation
  analyze_results.py         Post-hoc analysis and charts
  generate_rule_library.py   Compile rules.txt → rules.json
```


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
