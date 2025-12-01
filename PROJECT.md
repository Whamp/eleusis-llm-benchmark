# Eleusis LLM Benchmark project

This project is a benchmark for evaluating large language models (LLMs) on their ability to understand and generate rules in the context of the card game Eleusis.

For that we must:
- Implement the core game in Python, following the provided simplified ruleset, see RULES.md for details.
- Validate the implementation with unit tests.
- Create a framework so that LLMs can play the game by generating moves based on the game state. This involves defining prompts that guide the LLMs to make decisions according to the game's rules.
- Provide a testing framework to facilitate the assessment of LLMs' performance in playing the game, and rank them based on their success in understanding and applying the rules.

Open source LLM are called using Hugging Face Inference Providers API.

Closed source LLMs are called using the Openrouter API.