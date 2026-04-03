# Agent Instructions

## Project Overview

Eleusis LLM Benchmark — evaluates LLMs on inductive reasoning via a card game.
A secret rule governs which cards are accepted. The model discovers the rule by
playing cards and observing outcomes.

## Build & Run

- **Package manager:** `uv` (never `pip`)
- **Install:** `uv sync`
- **Lint:** `uv run ruff check`
- **Tests:** `uv run pytest`
- **Smoke test:** `uv run python scripts/evaluate_single.py --config config.smoke.yaml --model <key> --tag smoke`
- **Full benchmark:** `uv run python scripts/evaluate_single.py --model <key>`

## Architecture

Two separate model roles:

1. **Rule compiler** — converts natural-language rules to Python. Configured in `config.yaml`.
2. **Benchmark model** — plays the game. Selected via `--model <key>` from `models.yaml`.

Core modules:

- `src/eleusis/game/` — cards, game state, engine, rule validation
- `src/eleusis/llm/` — provider clients (Anthropic, OpenAI, Google, xAI, HuggingFace, OpenAI-compat)
- `src/eleusis/prompts/` — prompt templates
- `src/eleusis/analysis/` — results analysis and visualization
- `src/eleusis/runner.py` — round orchestration
- `src/eleusis/player.py` — LLMScientist player logic

Entry points:

- `scripts/evaluate_single.py` — main evaluation script
- `scripts/run_parallel_benchmark.py` — parallel multi-worker evaluation
- `scripts/run_parallel_eval.sh` — parallel multi-model evaluation
- `scripts/analyze_results.py` — post-hoc analysis
- `scripts/generate_rule_library.py` — compile rules.txt to rules.json

## Code Style

- Ruff for linting (line length 100, Python 3.11+)
- `snake_case` for functions/variables, `PascalCase` for classes

## Key Design Decisions

- Rule code runs in a restricted `exec()` sandbox (`engine.py`). Changes to `safe_globals` require careful review.
- Simulation-based rule comparison — guesses are validated by running simulations, not string matching.
- Double deck (104 cards). Hand size stays constant (draw after each play).
- Round seeds are deterministic: `base_seed + rule_hash + batch_round_index`.
- Each round of the same rule is independent — different `batch_round_index` values produce different deck shuffles.
- API keys go in `.env` (gitignored). Reference `.env.example` for the template.
