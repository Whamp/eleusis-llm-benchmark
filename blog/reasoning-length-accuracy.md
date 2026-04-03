# When Thinking Harder Makes You Wrong: Reasoning Length and Accuracy in the Eleusis Benchmark

*Running a local 27B model against frontier APIs revealed something the original benchmark missed: not all models benefit from longer reasoning chains.*

## Background

Hugging Face recently published the [Eleusis Benchmark](https://huggingface.co/spaces/huggingface/eleusis-benchmark), which evaluates LLMs as scientists. Based on the 1956 card game, a model must discover a secret rule governing which cards are accepted or rejected — by playing cards, observing outcomes, forming hypotheses, and deciding when to commit to a guess.

Their analysis identified two failure modes: **reckless guessing** (guessing too often, paying the -2 point penalty per wrong guess) and **excess caution** (knowing the answer but waiting too long, paying -1 per wasted turn). They showed models have distinct "scientist personalities" — GPT 5.2 Pro is extremely cautious, Grok 4.1 is reckless, and so on.

We ran the benchmark locally on a quantized Qwen3.5-27B-Opus-Distilled model (Q4_K_M, single RTX 3090 via llama.cpp) and dug into the per-turn data across all 13 models. We found a third dimension the original analysis didn't cover: **the relationship between reasoning length and accuracy, and how it differs dramatically across model families.**

## The Finding

We measured accuracy (shadow evaluation correctness) as a function of reasoning token count for every model in the benchmark. Three distinct patterns emerged.

### Pattern 1: Flat Thinkers (GPT 5.2 High)

```
GPT 5.2 High — accuracy by reasoning length:
  <500 tokens    36%  (28 guesses)
  500-1K         47%  (154 guesses)
  1K-2K          50%  (315 guesses)   ← peak
  2K-4K          46%  (220 guesses)
  4K-8K          40%  (107 guesses)
  8K-16K         37%  (30 guesses)
```

GPT 5.2 High maintains 36-50% accuracy regardless of how long it thinks. Short chains and long chains produce roughly equal quality hypotheses. This model genuinely uses extended reasoning productively. When it thinks longer, it's because the problem is harder — not because it's spiraling.

### Pattern 2: Overthinking Collapser (Claude Opus 4.5)

```
Claude Opus 4.5 — accuracy by reasoning length:
  500-1K         50%  (4 guesses)
  1K-2K          37%  (70 guesses)
  2K-4K          32%  (174 guesses)
  4K-8K          18%  (322 guesses)
  8K-16K          5%  (96 guesses)    ← collapse
```

Claude Opus 4.5 shows a monotonic decline. At under 2K tokens, it matches GPT 5.2's accuracy. But as reasoning chains grow past 4K tokens, accuracy collapses. At 8K-16K tokens, Claude gets the rule right only 5% of the time across 96 attempts. The model isn't thinking harder — it's overthinking, constructing increasingly baroque hypotheses that overfit the observed data.

### Pattern 3: Fast or Wrong (Qwen3.5-27B-Opus, DeepSeek R1, GPT-OSS 120B)

```
Qwen3.5-27B-Opus — accuracy by reasoning length:
  500-1K         31%  (102 guesses)   ← best range
  1K-2K          16%  (77 guesses)
  2K-4K           8%  (48 guesses)
  4K-8K          14%  (29 guesses)
  8K-16K         50%  (6 guesses)     ← too few to trust
```

These models show the steepest decline. Qwen drops from 31% accuracy under 1K tokens to 8% at 2K-4K. The sweet spot is narrow: short reasoning chains that identify a simple pattern. Everything else is noise.

DeepSeek R1 follows the same shape — 25% at 2K-4K, then 13% at 4K-8K, then 3% at 8K-16K. GPT-OSS 120B: 19% under 2K, then 3% at 4K-8K.

## The Full Picture

Across all 13 models tested on the benchmark:

| Model | Overall | <1K Accuracy | >4K Accuracy | Pattern |
|---|---|---|---|---|
| GPT 5.2 High | 46% | 45% | 40% | Flat |
| Claude Opus 4.5 | 22% | 50% | 15% | Collapser |
| GPT 5 Mini | 21% | 26% | 11% | Collapser |
| Qwen3.5-27B-Opus | 21% | 31% | 20% | Fast or wrong |
| Gemini 3 Flash High | 20% | — | 20% | Flat (but low) |
| GLM 4.7 | 19% | 15% | 18% | Flat (but low) |
| Kimi K2 | 18% | — | 16% | Flat (but low) |
| GPT-OSS 120B | 14% | 18% | 3% | Collapser |
| Grok 4.1 | 13% | 8% | 11% | Flat (but low) |
| DeepSeek R1 | 12% | — | 10% | Collapser |
| GPT-OSS 20B | 12% | 12% | 6% | Collapser |
| Claude Haiku 4.5 | 9% | — | 7% | Collapser |

The column that matters most is the delta between "<1K Accuracy" and ">4K Accuracy." For GPT 5.2 High, it's 5 percentage points (45→40). For Claude Opus, it's 35 points (50→15). For GPT-OSS 120B, it's 15 points (18→3).

## Why This Happens

The Eleusis benchmark exposes a specific failure mode: **Occam's Razor violations under uncertainty.** When a model encounters ambiguous evidence — say, three clubs accepted and one spade rejected — a simple hypothesis ("clubs only") explains everything. A model that reasons briefly states this and moves on. A model that reasons at length starts considering alternatives: "maybe it's clubs and cards with even ranks, or clubs and cards where the rank exceeds the previous card's rank by at most 3."

We confirmed this directly. In our Qwen model's data, wrong guesses averaged 1.9x the AST node complexity of correct guesses. The model's incorrect hypotheses weren't just wrong — they were structurally more complex. It invented conditional rules, multi-attribute filters, and relational constraints when the actual rule was a single attribute check.

This is consistent with overfitting in a low-data regime. The model treats its chain-of-thought as a search process, and longer searches explore more of the hypothesis space. But the Eleusis game provides sparse evidence (typically 3-10 data points per hypothesis), so complex hypotheses that fit the training data rarely generalize.

## Implications

**For model developers:** The overthinking collapse suggests that extended reasoning capabilities don't automatically transfer to inductive reasoning tasks. GPT 5.2 High's flat curve is the exception, not the rule. Most models would benefit from mechanisms that detect when reasoning chains are growing without converging — and terminate early.

**For practitioners using local models:** If you're running a quantized reasoning model on consumer hardware, longer generation limits won't compensate for architectural limitations. Our Qwen model's accuracy peaked under 1,000 reasoning tokens. Setting `max_tokens` to 32,768 wouldn't have helped — it would have produced longer wrong answers.

**For benchmark design:** The Eleusis benchmark's structured output format (requiring a tentative rule, confidence level, and reasoning at every turn) makes this analysis possible. Most benchmarks report only final accuracy. The per-turn data here reveals metacognitive patterns invisible in aggregate scores.

## A Note on Running Eleusis Locally

We ran the benchmark on a Qwen3.5-27B-Opus-Distilled model (Q4_K_M quantization) served by llama.cpp on a single RTX 3090, using an OpenAI-compatible API provider. The model completed 28 of 78 rounds over about 11 hours, going 28W/0L with an average score of 17.1 (no-stakes: 22.0).

The benchmark works well with local models — the only friction was writing a thin provider to handle llama.cpp's `reasoning_content` field format and setting timeouts high enough for a ~31 tok/s server.

The full benchmark code, our provider, and analysis scripts are available in the [Eleusis LLM Benchmark repository](https://github.com/huggingface/eleusis-llm-benchmark).

---

*Data: 13 models evaluated on the Eleusis benchmark. Frontier model results from Hugging Face's 78-round evaluation (January 2026). Local model results from our 28-round evaluation (March 2026). Analysis covers 11,797 total hypothesis evaluations across all models.*
