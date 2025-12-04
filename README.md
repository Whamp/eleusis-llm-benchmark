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

# Play a single round with default config
uv run scripts/play_single_round.py

# Play a tournament (multiple rounds)
uv run scripts/play_tournament.py
```