# Eleusis LLM Benchmark

This project is a benchmark for evaluating large language models (LLMs) on their ability to understand and generate rules in the context of the card game Eleusis. 

The core game is implemented in Python, following the provided simplified ruleset, see RULES.md for details.

A testing framework is provided to facilitate the assessment of LLMs' performance in playing the game.

Open source LLM are called using Hugging Face Inference Providers API.

Closed source LLMs are called using their respective APIs.

## Generate a library of rules

To generate a library of rules, run:

```bash
uv run scripts/generate_rule_library.py --num-rules 50 --output rules.json --model openai/gpt-oss-120b --max-tokens 16384 --test-cases 5
```

Evaluate the rules generated:

```bash
uv run scripts/evaluate_rules.py --library rules.json
```
