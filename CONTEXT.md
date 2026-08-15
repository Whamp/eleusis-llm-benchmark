# Eleusis LLM Benchmark

This project evaluates how benchmark models infer secret Eleusis rules from card-play feedback.

## Language

**Benchmark Run**:
One resolved benchmark configuration executing an immutable ordered schedule of Rounds under fixed scientific settings and source behavior.
_Avoid_: Evaluation, experiment

**Round**:
A complete Eleusis game played from fresh game state under one secret rule. A round ends with a terminal outcome.
_Avoid_: Episode

**Episode**:
A future reinforcement-learning interaction that may wrap a Round. It is not a synonym for a Round in the current benchmark domain.

**Round Record**:
The authoritative, structured account of one Round and its ordered Turns. Once the Round reaches a Terminal Outcome, its Round Record is immutable; scores, analyses, and future training data are derived from it.
_Avoid_: Round result, round log, results dictionary

**Round Checkpoint**:
A durable continuation point for an in-progress Round after setup or a completed Turn. It combines the active Round Record with the stable game state needed to begin the next Model Attempt.

**Turn**:
One model decision cycle within a Round: the state presented to the benchmark model, its Model Attempts, the resulting card play and replacement draw, and any optional Guess Attempt.

**Model Attempt**:
One prompt submission and interpretation cycle used to obtain a model decision during a Turn. It includes the prompt, textual completion when available, interpretation outcome, nested Provider Calls, and any error or retry cause; changing the retry prompt begins another Model Attempt.

**Provider Call**:
One observable API request and response made within a Model Attempt or Rule Compilation Attempt. Continuations and visible transport retries are separate Provider Calls.

**Fallback Decision**:
A card decision made by the benchmark after its Model Attempts fail to produce a usable decision. It records its fallback origin rather than masquerading as a model response.

**Card**:
A playing card identified by rank and suit. Identical cards from the two decks are interchangeable; their multiplicity and order are represented by the collections containing them.

**Rule Compilation Attempt**:
One generation, extraction, and validation cycle used to compile a natural-language Guess Attempt into executable rule code. It preserves its nested Provider Calls, validation failures, simulation settings, and verdict evidence; a compiler cache hit instead records the reused artifact and its provenance.

**Guess Attempt**:
A proposed rule evaluated against the secret rule during a Turn. A Guess Attempt is either a Formal Guess or a Shadow Guess.
_Avoid_: Guess

**Formal Guess**:
A Guess Attempt that affects gameplay and scoring.

**Shadow Guess**:
A counterfactual Guess Attempt recorded without changing gameplay or scoring.

**Shadow Verdict**:
A versioned, post-hoc evaluation of a Shadow Guess by an identified judge. Shadow Verdicts are attached to the immutable trajectory rather than incorporated into it.

**Terminal Outcome**:
The conclusion of a Round: a correct Formal Guess, reaching the turn limit, or an explicit decision that continuation is impossible or the Round is abandoned. A process interruption is not a Terminal Outcome.

**Record Invalidation**:
An annotation declaring that an immutable Round Record must not be treated as valid scientific evidence. It preserves the original trajectory rather than rewriting it.
