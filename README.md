# Eleusis LLM Benchmark

This project is a benchmark for evaluating large language models (LLMs) on their ability to understand and generate rules in the context of the card game Eleusis. 

The core game is implemented in Python, following the provided simplified ruleset, see RULES.md for details.

A testing framework is provided to facilitate the assessment of LLMs' performance in playing the game.

Open source LLM are called using Hugging Face Inference Providers API.

Closed source LLMs are called using their respective APIs.