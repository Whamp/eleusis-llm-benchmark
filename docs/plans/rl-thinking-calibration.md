# Plan: Eleusis as an RL environment for thinking calibration

Status: proposal. This is not an accepted ADR. It must not silently override
[ADR 0001](../adr/0001-use-per-run-sqlite-as-authoritative-store.md),
[ADR 0002](../adr/0002-keep-completed-round-records-immutable.md), or
[ADR 0003](../adr/0003-refuse-resume-under-changed-scientific-provenance.md).

Domain terms follow `CONTEXT.md`. An Episode may wrap a Round; it is not a
synonym for a Round.

## Goal

Train a policy whose *thinking* is calibrated on inductive tasks. The useful RL
target is not a higher Eleusis score. Score already mixes discovery, speed, and
guess penalties. What this environment can uniquely train is when thinking is
warranted, when a hypothesis is actually known, and when to commit.

That is three calibration problems the current models split apart:

- GPT-5.2 High knows, waits, and under-commits.
- DeepSeek V4 Flash and DeepSeek R1 think hard, commit hard, and often have no
  correct Shadow Guess before the Formal Guess.
- Claude Opus 4.5, Qwen, and GPT-OSS often get worse as Reasoning Traces get
  longer.

Success means all of the following, with hold-out Eleusis score treated as a
constraint rather than the objective:

1. Stated confidence matches actual Shadow Verdict correctness.
2. A Formal Guess happens when the Shadow Guess is already correct, not long
   before or long after.
3. Extra reasoning tokens raise Shadow accuracy, or the model stops; they must
   not collapse it.
4. Hold-out Eleusis score stays in the Opus band.

Keep the published 26-rule leaderboard as an eval-only lane. Training Episodes
must not reuse those secret rules.

## Why this environment

A Turn already emits the labels RL needs: card accept/reject, a
confidence-scored tentative rule, an optional costly commitment, a Reasoning
Trace, and (offline) a Shadow Verdict that does not change the game.

The existing analyses are the eval suite: calibration curves, excess caution,
reckless guessing, complexity ratio, tokens-by-turn, no-stakes score. Those
curves moving is how we know training worked.

The current score function is a bad primary RL reward. It would push V4 Flash’s
personality (solve fast, guess aggressively) or GPT-5.2’s (never be wrong, bleed
turns), depending on hyperparameters. Neither is calibrated thinking.

Eleusis is a better env than a generic reasoner gym because every Turn is
already a structured action:

- play a Card (information)
- optional Shadow Guess plus confidence 0–10 (belief)
- optional Formal Guess (commitment; −2 if wrong)
- a Reasoning Trace (the thing to regularize)

## Environment design

Treat Episode as a wrapper around one Round, exactly as `CONTEXT.md` already
reserved. Do not rename Round.

```text
Episode
  └─ Round (fresh deck, one secret rule, immutable Round Record)
       └─ Turn t = 1..T
            observation: board, hand, history, remaining turns, prior failed Formal Guesses
            latent: Reasoning Trace
            action: card + optional tentative rule + confidence + optional Formal Guess
            env: accept/reject, replacement draw
            labels: Shadow Verdict (process), Formal outcome (commitment), score delta
```

Hard constraints from the ADRs:

- Completed Round Records stay immutable. Training tuples are derived datasets.
- Secret rule never enters the observation.
- Rule Compilation Fidelity is apparatus. A noisy compiler is reward noise.
  Freeze compiler identity for a training lane, or judge structured hypotheses
  against the secret Python rule and keep NL compilation for transfer eval only.
- Fallback Decisions are not policy actions. Drop them from training.

### Action space

In order of cleanliness:

1. **Start structured.** The agent outputs a card plus a predicate over
   card/history features the env can execute (color, suit, rank, parity,
   face/number, last-card relations). Immediate, exact Shadow label. This is the
   training env.
2. **Then NL transfer.** Same policy, but the tentative rule is natural
   language, judged by the frozen compiler. This is the transfer env and matches
   `LLMScientist`.
3. Do not start with free-form NL plus a live LLM compiler. That is too slow and
   too noisy for a first RL loop.

### Rule generator for training

Do not train on `rules.json`. Sample compositional secret rules from the same
feature algebra (unary predicates, last-card relations, simple alternation,
bounded AND/OR). Hold out entire families for eval (for example all alternation
rules, or all face-card modifiers), not just random instances. Keep
`screen_26x1` / `full_26x3` as frozen benchmark lanes.

## What to optimize

Use a vector of rewards, logged separately, combined only after each term can be
plotted.

**Belief calibration (primary).** For every Shadow Guess with confidence
`c ∈ [0, 10]` and Shadow Verdict `y ∈ {0, 1}`:

```text
r_cal = -((c / 10) - y)^2
```

This directly attacks “90% confident, 22% correct.”

**Commitment timing (primary).** At Formal Guess time:

- correct Formal Guess and latest Shadow was already correct: small bonus (it
  committed what it knew)
- correct Formal Guess with no prior correct Shadow: smaller bonus or zero
  (lucky or unverbalized knowledge)
- wrong Formal Guess: the existing −2, plus extra if `c` was high
- correct Shadow exists and the agent still does not Formal Guess: per-turn
  caution penalty, capped, so it cannot be farmed by never committing

That is excess caution vs reckless guessing as an actual training signal, not
just a chart.

**Thinking utility (primary for calibrating thinking).** Conditional on the
Shadow Verdict:

- if `y = 1`, no length penalty (or a very weak one)
- if `y = 0`, penalty that grows with reasoning tokens after a short budget
- optional stop-thinking bonus: producing a correct Shadow below the model’s
  historical collapse threshold

This is aimed at Opus/Qwen/R1’s overthinking collapse and at V4 Flash’s
~23k-token Turns that still have weak shadows.

**Information value of the card (secondary).** Reward playing a card that splits
remaining hypotheses, or at least do not reward repeating a test the current
Shadow already predicts. Otherwise the policy can stall on safe cards while
“thinking.”

**Terminal score (constraint).** Floored Round score as a floor: reject
checkpoints that collapse success rate on hold-out rules, even if calibration
looks pretty.

Do not reward “more reasoning” or “higher confidence.” Those are the failure
modes.

## Learning methods, in order

### Phase A — Offline dataset from existing Round Records

Existing corpus: historical 78-round runs, Luna, DeepSeek V4 Flash, Qwen, and
later runs. Derive per-Turn tuples

```text
(observation, reasoning, card, shadow, confidence, formal, verdict, tokens)
```

without rewriting records. Split by model personality so training can use:

- a cautious expert (GPT-5.2 traces)
- an aggressive expert (V4 Flash / R1 traces)
- a calibration filter (keep traces where confidence ≈ shadow accuracy)

This is cheaper than online LLM rollouts and lets rewards be debugged against
known curves.

### Phase B — Behavior clone the process, not the score

SFT/DPO on:

- Shadow text that later received a correct Verdict, vs plausible wrong shadows
  from the same Turn
- Formal Guess on the first Turn with a correct Shadow, vs later Turns
  (anti-caution)
- short correct traces vs long incorrect traces from the same model
  (anti-collapse)

This should move calibration before any policy-gradient noise.

### Phase C — Offline RL on the derived dataset

IQL/CQL or token-level DPO. Optimize `r_cal + r_commit + r_think`. Keep the
Eleusis score as a conservative Q-constraint.

### Phase D — Online RL in the structured env

GRPO/PPO with a frozen secret-rule judge, short Rounds (max 15–20 Turns), cheap
models first (the 27B class that can actually be iterated). Online shadows every
Turn with confidence ≥ 5, or every Turn if that gate is dropped for training.
The current `confidence < 5` skip is an eval convention; for RL a belief is
wanted every Turn.

### Phase E — NL transfer

Wrap `LLMScientist` so one Episode is one Round, compiler frozen,
`shadow_mode: online` for training only. Eval with `shadow_mode: offline` plus
`scripts/evaluate_shadows.py` so training judges cannot leak into the published
measurement.

## Repo work, in this order

1. **Episode adapter, no training yet.** A Gymnasium-style env that starts a
   Round, steps a Turn, returns observation/action/reward fields, and writes
   Round Records as today. Training code must not live inside `runner.py`.
   Issue #13 (deep round-execution module) is the right seam.

2. **Train/eval lane split.** Generated training rules vs frozen `rules.json`.
   Issues #5 and #9 (manifest and lane-aware analysis) should gate “this run is
   RL-train” vs “this run is benchmark.” Never mix them in
   `results/260312_all_models_corrected`.

3. **Dense structured judge.** Execute hypothesized predicates against the
   secret rule on a fixed simulation pack (the same 100×40 idea as Shadow
   Verdicts, but no LLM). This unblocks Phase D.

4. **Trajectory export.** A derived dataset format: one row per Turn from
   SQLite Round Records + Shadow Verdict sidecars. Read-only over existing
   results.

5. **Calibration trainer.** Offline SFT/DPO first, then RL. Success is plots,
   not loss: calibration curve closer to `y = x`, excess-caution mean moving
   toward ~1–2 on wins, double-down falling from ~50% without success
   collapsing, complexity ratio nearer 1, tokens-by-turn not exploding on wrong
   shadows.

6. **Hold-out eval protocol.** After each training checkpoint, run `screen_26x1`
   on a frozen model key that is not in the training rule generator, plus the
   published 26 if a public number is wanted. Report the existing analysis
   folder, not a custom score.

Issue #7 (shadow replay matching online judging) and #12 (one verdict path)
become blocking for Phase E, not for Phase A–D. Fix the judge before NL online
RL, not before structured offline work.

## Failure modes

- **Reward hacking the compiler.** The policy learns phrases the compiler maps
  to the secret rule without the model understanding it. Frozen compiler +
  simulation judge + structured-first training is the mitigation.
- **Benchmark contamination.** Training on the 26 published rules makes the
  leaderboard meaningless. Generated rules and family holdouts are the
  mitigation.
- **Score hacking.** Maximizing floored score recreates V4 Flash. Keep score as
  a constraint.
- **Thinking collapse the other way.** A naive length penalty produces
  short confident wrongness. Length penalty must be conditional on a wrong
  Shadow.
- **Mute shadows.** Policy stops emitting Shadow Guesses to avoid calibration
  loss. Require a tentative rule every Turn during training.
- **Using Fallback Decisions as expert actions.** They are random cards.
  Exclude them.

## First experiment

Not online PPO on V4 Flash. Too expensive, too many tokens, and its shadows are
already sparse (15/104 correct in the `screen_26x1` run).

Run this instead, on a local 27B:

1. Export Turn tuples from Qwen, R1, V4 Flash, and GPT-5.2.
2. DPO: correct short shadows vs incorrect long shadows; commit on first
   correct shadow vs delayed commit.
3. Eval on a held-out generated 12-rule set and on `stress_12x1`.
4. Look at one plot first: shadow accuracy vs stated confidence. If that curve
   does not move, the later RL stack will not either.

If that curve moves and `stress_12x1` success does not crash, then stand up the
Episode env and do GRPO on structured hypotheses. Only then wrap `LLMScientist`
for NL transfer.

The scientific claim this would support: Eleusis can train metacognition of
inductive reasoning — knowing when a hypothesis is done — rather than merely
training models to grind a higher card-game score.
