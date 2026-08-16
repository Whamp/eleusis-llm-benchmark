# Rule Compiler Tournament — 2026-08-16

## Why we ran this

The benchmark's rule compiler turns written rule descriptions into Python code. We
needed to pick which company's servers should run it through OpenRouter, and compare
gpt-oss-120b (the incumbent model) against DeepSeek v4 Flash (a challenger).

OpenRouter's published claims about what each server supports proved unreliable, so
everything here was decided by direct testing.

## Step 1: Which servers honor the temperature setting?

Method: same prompt sent many times at temperature 0 (answers should be identical) and
temperature 1 and 2 (answers should spread out). If answers vary at temperature 0, or
fail to spread as temperature rises, the server is ignoring the setting.

| Server | Model | Verdict |
|---|---|---|
| Cloudflare | deepseek-v4-flash-0731 | Honors temperature cleanly |
| DeepSeek (first-party) | deepseek-v4-flash-0731 | Ignores it — claimed support, doesn't deliver |
| Cerebras | gpt-oss-120b | Honors |
| Nebius | gpt-oss-120b | Honors |
| Novita | gpt-oss-120b | Honors |
| AkashML | gpt-oss-120b | Honors |
| Groq | gpt-oss-120b | Honors (weakly) |
| Together, DeepInfra, SiliconFlow, CoreWeave | gpt-oss-120b | Fail — excluded |
| Fireworks | deepseek-v4-flash-0731 | Unavailable (persistent 429s) |

## Step 2: The tournament

Every qualifying server compiled all 26 benchmark rules, 3 times each, at reasoning
effort low, medium, and high (1,404 calls total, ~$4). Each result was scored by the
production oracle: does the compiled code behave identically to the known-correct rule
across 100 seeded simulations of 40 turns?

Ten of eighteen cells scored a perfect 78/78. Full data: `results.json`. Harness:
`harness.py`.

Key findings:

- **Novita (gpt-oss-120b) is perfect at all three effort levels and tied-cheapest.**
  It also demonstrably obeys the effort dial (mean thinking tokens 226 → 238 → 697
  from low to high).
- AkashML, Nebius, and Groq are also perfect at low and/or medium.
- **DeepSeek v4 Flash eliminated**: at its best (low effort) it merely ties gpt-oss
  while costing ~10× more per conversion, and it got *worse* with more thinking —
  long reasoning hit the output limit and produced broken code at higher efforts.

## Decision

- **Primary compiler: gpt-oss-120b on Novita, effort medium, temperature 0.7.**
- **Fallback waterfall** (used only if the primary fails after retries), each pinned
  to its server so routing is deterministic and reproducible:
  1. Novita (primary)
  2. AkashML
  3. Nebius
  4. Groq
  5. Cerebras
  6. Cloudflare deepseek-v4-flash-0731 (different model entirely — last-resort
     diversity against a gpt-oss-wide outage; run at temperature 1.0, effort low)

## Future option: self-hosted DeepSeek

Will runs DeepSeek v4 Flash on his own server. If the full benchmark (compiling messy
model-written guesses rather than clean rule descriptions) proves harder than this
tournament, the self-hosted server is a free way to test whether v4 Flash's extra
capability helps. Server was down during this evaluation; untested, no OpenRouter cost.

## Caveats

- Tournament input was clean rule descriptions from `rules.json`. Production compiles
  messier, model-written guesses; relative rankings may shift. Re-run `harness.py`
  against recorded guesses if compiler errors appear in real runs.
- Scores of 100% mean this task saturates; it separates price, not quality.
