I've completed the code investigation as a scout and updated the context file.

### Findings Summary
- **Core Benchmark**: Evaluates LLMs on inductive reasoning using the Eleusis card game.
- **Recent Update**: Integrated `openai_compat` provider to support self-hosted models like **Qwen3 (27B GPTQ)** and **DeepSeek R1**, specifically capturing their internal reasoning/thinking steps.
- **Architecture**:
    - **Environment (`GameEngine`)**: Enforces secret Python-based rules in a safe sandbox.
    - **Scientist (`LLMScientist`)**: The AI agent making plays and guessing rules.
    - **Verification (`RuleValidator`)**: Uses a "rule compiler" LLM to turn natural language guesses into code, then runs parallel simulations of both the actual and guessed rules to detect logical mismatches.
- **Model Config**: Centralized in `models.yaml`, where the new `qwen3.5-27b-gptq-int4` is defined with its endpoint and `reasoning_format`.

The full structured findings, including key code snippets and architecture details, are now available in `/home/will/benchmark/eleusis-llm-benchmark/context.md`.