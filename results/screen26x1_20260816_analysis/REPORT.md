# screen_26x1 Smoke Run — Analysis Report

**Run family:** `solo_evaluation_20260816_210723_screen26x1_w{0..12}` · started 2026-08-16
**Benchmark model:** `gpt-5.6-luna-deepthink` (ChatGPT Codex subscription via pi auth, effort `none`, deep-think scratchpad extraction)
**Rule compiler:** Novita `openai/gpt-oss-120b` (medium) via OpenRouter, fallback waterfall
**Code provenance:** workers pinned to revision `fb6a607`; shadow evaluation and this report produced at `cfd7972` + the two analysis fixes below
**Purpose:** first end-to-end pipeline exercise — gameplay, Round Records, crash-resume, parallel workers, offline Shadow Guess evaluation, and analysis reporting — ahead of the real study.

## Scope

Of 26 scheduled rounds (13 workers × 2), **21 completed** and are analyzed here.
Five workers (w5, w6, w10, w11, w12) are parked mid-round-2 with valid checkpoints:

- Round 2 was never finishable under the pinned revision: the Codex subscription
  backend deterministically moderation-flags the deep-think extraction prompt on
  full-length game payloads (w11 confirmed 2/2 old framing, 4/4 neutral framing;
  earlier "passing" probes used a truncated payload and do not generalize).
- Retrying identical flagged prompts only manufactured random fallback turns at
  3× call cost, so the five rounds were abandoned rather than ground through.
- The parked round-2s hold 29 fallback turns out of 76 (38%) — they are not
  usable scientific data and are excluded from every number below.
- Round 1 evidence says parking them loses little: all four rules that reached
  `turn_limit` in round 1 belong to these workers (see Shadow Findings), i.e.
  the model was genuinely stuck on the hard rules, not secretly close.

**Reasoning condition status:** deep-think full-trace extraction is
provider-constrained on this backend and is not pursued further; native
reasoning summaries were never part of this run. Every analyzed turn's
reasoning evidence is the deep-think scratchpad text as captured at runtime.

## Headline Results (21 completed rounds)

| Metric | Value |
| --- | --- |
| Rounds won by formal guess | 17 / 21 (81.0%) |
| Rounds at turn limit | 4 / 21 (all four hardest rules) |
| Total / average score | 311 / 14.8 |
| Total turns | 318 |
| Total output tokens | 106,135 (~336 / turn) |
| Output-token trend, early → late turns | 303 → 154 (−49.3%) |
| Wrong formal guesses | 7 (max streak 2; double-down rate 28.6%) |
| Complexity → success correlation | −0.552 (Q1–Q2 quartiles 100% win, Q3–Q4 60%) |

Full tables and charts: `summary.txt`, `basic_metrics.csv`, `*.png` / `*.json`
in this folder.

## Shadow Findings (159 offline verdicts)

Every shadow guess in the 21 completed rounds was compiled and compared
(100 simulations × 40 turns, seed 42). No compilation failures; verdicts are
either genuine equivalence (208,000 matched comparisons) or a first mismatch.

- **55 / 159 shadow guesses (34.6%) were logically equivalent to the secret rule.**
- In 12 of the 17 won rounds the model produced a *correct shadow guess
  before its winning formal guess* (median lead 3.5 turns, range 1–17;
  e.g. w7-r2 turn 6 → won turn 23). The model often knew the rule tentatively
  before committing.
- In 5 won rounds the formal guess succeeded with no prior correct shadow —
  commitment and tentative belief genuinely diverged there. (17 won = 12 with
  prior correct shadow + 5 without.)
- **All four turn-limit rounds show zero correct shadow guesses in 30 turns**
  (w5 "only black face cards", w6 "face cards red / number cards black",
  w10 "heart/spade vs club/diamond group alternation", w11 "red ⇒ rank up,
  black ⇒ rank down"). The model did not know these rules and never nearly
  did. These rules also carry the highest complexity (Q3/Q4 quartiles).

This is exactly the calibration signal the benchmark exists to measure:
tentative knowledge and committed action are distinguishable, and the
distinction tracks rule difficulty.

## Disclosures

1. **Fallback turns.** 8 of 318 analyzed turns (2.5%) ended in a random
   fallback card (`final_decision.origin = "fallback"`, cause
   `retry_exhausted`) after provider-capacity aborts — concentrated in the
   three hardest rules (w6-r1: 3, w10-r1: 4, w11-r1: 1). They are honestly
   recorded, filterable, and included in scoring as played.
2. **Parked rounds.** The five partial round-2s (76 turns, 29 fallbacks,
   22 unevaluated shadow guesses) are excluded from all numbers above.
3. **Moderation storm.** The fallback concentration is a downstream effect of
   the deterministic moderation flag described under Scope; identical-prompt
   retries cannot succeed against it. Product follow-up filed: moderation
   exhaustion should abort-and-stay-resumable instead of playing fallbacks.
4. **Shadow-eval transport.** OpenRouter returned transient 429s (shared
   Novita pool) during evaluation; all recovered within the compiler retry
   budget. Zero verdicts affected.

## Pipeline Exercise Findings

Validated end-to-end: strict Round Record persistence and resume (including a
genuine crash-resume drill), 13-worker parallel execution with per-worker
seeding, offline Shadow Verdict sidecars with judge-identity provenance,
SQLite-authoritative analysis views, and full report generation.

Two defects surfaced by this run, fixed test-first (226 tests green):

1. **Strict analysis views lacked rule complexity metrics.** Legacy
   `results.json` embedded `node_count` / `cyclomatic_complexity` per rule;
   strict Round Records persist only the rule code, and the analysis view
   didn't recompute them. Complexity analysis now derives them from the
   scheduled rule code (`benchmark_run_artifact.py`), and
   `analyze_complexity` skips binning with an explicit note when no
   complexity data exists instead of raising `KeyError`
   (`complexity.py`). Tests:
   `test_strict_analysis_view_computes_rule_complexity_metrics`,
   `test_analyze_complexity_skips_binning_when_metrics_absent`.

## Artifacts

- Run stores: `results/solo_evaluation_20260816_210723_screen26x1_w{0..12}/`
  (symlinked from this folder; `benchmark_run.sqlite3` authoritative,
  `results.json` regenerated with shadow verdicts)
- This report, `summary.txt`, charts and JSON sidecars in this folder
- Supervisor/ops scripts under `/tmp` were not persisted (intentional; run is
  complete)
