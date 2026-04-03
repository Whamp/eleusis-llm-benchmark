# Suggested AGENTS.md Additions for Qwen3.5 27B

Based on failure patterns identified in the Eleusis LLM Benchmark (inductive
reasoning / scientific hypothesis testing). Each rule maps to a specific failure
mode observed across 78 rounds of evaluation, with design inspiration from MARL's
multi-stage verification pipeline (S1 Hypothesis → S3 Auditor → S4 Verifier).

These go under `## Reasoning Discipline` in your AGENTS.md.

---

## Reasoning Discipline

### Hypothesis diversity (fixes: anchoring, blind spots)

_Eleusis failure: Qwen locked onto "same color" from turn 1 and spent 31 turns
patching it instead of considering "face vs number" — a simpler, correct rule._

_MARL origin: S1_Hypothesis — "divergent search: core trap, key contradiction,
best angle."_

```
- List multiple hypotheses before evaluating. False confidence exists.
- Before committing to an explanation, list at least 3 alternatives from
  different categories. Include at least one you consider unlikely.
```

### Occam's razor enforcement (fixes: overcomplication)

_Eleusis failure: 79% of wrong guesses were more complex than the actual rule.
Model added "black + multiple of 4 + rank diff of 8" when the rule was just
"alternate face and number cards."_

_MARL origin: S3_Auditor — "overconfident claims, domain drift."_

```
- Prefer the simplest explanation that fits all observations. If a complex
  hypothesis and a simpler one both explain the data, choose the simpler one.
- When a solution requires special-case logic, compound conditions, or multi-step
  patterns, treat that as a signal that you may be overfitting to noise rather
  than identifying the actual pattern.
```

### Confidence recalibration (fixes: 22% accuracy at self-reported 90%)

_Eleusis failure: At confidence 9 ("90% sure"), Qwen was correct 21.8% of the
time. At confidence 10, just 46.4%. GPT 5.2 High made 22 wrong guesses total
across all 78 rounds; Qwen made 267._

_MARL origin: S4_Verifier — "[TRAP-CHECK] hidden traps? Y/N. [HALLUCINATION]
unverifiable claims? Y/N."_

```
- Your confidence is likely miscalibrated. When you believe you are 80-90%
  confident, your actual accuracy is closer to 15-20%. Treat your own high
  confidence as a weak signal, not a strong one. Require concrete, falsifying
  evidence — not just pattern-matching — before committing to a conclusion.
```

### Periodic hypothesis reset (fixes: incremental patching)

_Eleusis failure: Qwen never resets after failure. "Same color" → "same color AND
parity" → "same color AND parity AND lower rank" → "black AND multiple of 4." Each
step adds complexity to a wrong base rather than questioning it._

_MARL origin: S2_Solver — "[BACKTRACK] I adjust X because Y. Corrected: Z."_

```
- After a failed approach, do not retry with a minor variation. Stop, reassess
  your assumptions, and verify the foundational premise is correct before trying
  again.
- After a failed approach, stop and reassess before trying again. Do not
  immediately retry a variation of the same idea. Instead, list what the failure
  revealed, identify which of your assumptions it invalidated, and only then form
  a new hypothesis.
```

### Adversarial self-check (fixes: OR-instability, false exceptions)

_Eleusis failure: Qwen found the correct rule "same color OR same parity" at turn
15, then lost it by adding "except face cards" / "except Kings." It oscillated
between the right answer and overcomplicated variants for 11 more turns._

_MARL origin: S4_Verifier — adversarial cross-validation (S1→S3, S2→S4, etc.)._

```
- When confident in a solution, actively look for the simplest counterexample
  that would disprove it before committing.
- When multiple observations fit your hypothesis, actively seek the one
  observation that would break it. If you cannot find a counterexample after
  deliberate effort, then you have earned the right to commit.
```

### Simplest generalization (fixes: overfitting to observed data)

_Eleusis failure: Qwen would construct rules that perfectly fit the 5-10 observed
cards but used card-specific conditions ("rank must be 4, 5, 12, or 13") instead
of the underlying pattern ("alternate face and number")._

_MARL origin: S3_Auditor — consistency gate, drift detection._

```
- Prefer the most general explanation that fits all observations over a specific
  one that fits only what you've seen. Overfitting to observed data produces
  brittle solutions.
```

---

## Notes

Several of these rules already appear in our global AGENTS.md — they were written
based on exactly these failure patterns observed during the Eleusis benchmark
evaluation. The MARL pipeline's separation of concerns (hypothesis generation →
consistency auditing → adversarial verification → refinement) maps naturally to
the cognitive failures: each stage catches a different class of reasoning error.

The key insight from both the benchmark and MARL: **a single-pass reasoner that
generates and evaluates in the same breath will anchor on its first hypothesis.**
Separating hypothesis generation from hypothesis evaluation — even within a single
prompt via structured sections — is the core fix.
