# Qwen3.5 27B GPTQ Int4 — Eleusis Benchmark Tweet Thread

## Tweet 1 — Hook + Leaderboard
**Image:** `/tmp/tweet-images/1-leaderboard.png`

I ran a quantized 27B model on a single RTX 3090 against 15 frontier LLMs on the Eleusis benchmark — a card game that tests scientific reasoning.

Qwen3.5 27B GPTQ Int4 scored 15.1, tying GPT 5.2 High.

Here's what I found 🧵

**[225 chars]**

---

## Tweet 2 — Key Stats
**Image:** `/tmp/tweet-images/2-stats.png`

2/ The key numbers:

Score: 15.1 (ties GPT 5.2 High)
No-stakes score: 19.1 (≈ GPT 5.2's 19.3)
Success: 69/78 rounds solved
Wrong guesses: 3.4/round (optimal: ~0.5)

Same raw discovery ability as a frontier model. Radically different self-awareness.

**[248 chars]**

---

## Tweet 3 — Scientist Personality
**Image:** `/tmp/tweet-images/3-personality.png`

3/ The paper classifies models as "scientist personalities."

Qwen is a reckless scientist — guesses too often, gets punished.

GPT 5.2 is cautious — knows the answer but waits too long.

Claude Opus is balanced.

The bottleneck isn't reasoning. It's metacognition.

**[265 chars]**

---

## Tweet 4 — Rule Complexity Breakdown
**Image:** `/tmp/tweet-images/4-complexity.png`

4/ Where it dominates vs collapses:

Simple rules (is it red? even? a face card?): 100% success. Best of ALL 15 models.

Complex rules (alternate face/number, non-standard suit groups, OR-conditions): 0 out of 9 rounds. Dead last.

**[230 chars]**

---

## Tweet 5 — The Anchoring Failure
**Image:** `/tmp/tweet-images/5-anchoring.png`

5/ Most revealing failure: "alternate face and number cards."

Qwen locks onto "same color" from turn 1. Over 31 turns it tries same parity, color+parity, multiples of 4, rank diff of 8...

Never once hypothesizes face vs number. 13 wrong guesses.

**[247 chars]**

---

## Tweet 6 — Calibration + Takeaway
**Image:** `/tmp/tweet-images/6-calibration.png`

6/ When Qwen says "90% confident" it's right 22% of the time. At "100%" — just 46%.

GPT 5.2 High made 22 wrong guesses total. Qwen made 267.

For local models to compete, they don't need better reasoning — they need to know when they don't know.

**[246 chars]**

---

## Tweet 7 — AGENTS.md Fixes
**Image:** `/tmp/tweet-images/7-agents.png`

7/ So how do you fix this with prompting?

We mapped each failure to an AGENTS.md rule, inspired by MARL's multi-stage verification pipeline (S1 Hypothesis → S3 Auditor → S4 Verifier).

6 targeted patches. All under "Reasoning Discipline."

Full additions below:

**[262 chars]**

---

## Tweet 8 — Copy-Paste AGENTS.md Rules
**Image:** `/tmp/tweet-images/8-copypaste.png`

8/ Copy-paste these into your system prompt.

Each maps to a failure we measured:
- Anchoring → hypothesis diversity
- Overcomplication → Occam's razor
- Miscalibration → falsification gate
- Patching → forced reset
- Instability → adversarial self-check

**[254 chars]**
