# Round resume checkpoint prototype

## Question

Can an Eleusis Round cross a structured JSON and process boundary after setup or a completed Turn, then expose the same next model prompt and produce the same next transition as uninterrupted execution?

## Verdict

Yes. The current runtime lacks restoration APIs, but its continuation state is bounded and serializable. The prototype restored fresh `Deck`, `GameState`, `GameEngine`, `LLMScientist`, model-client, compiler-cache, and validator-cache objects without calling `setup_game()` again.

Bounded does not mean trivial. The throwaway script is 695 lines, including its subprocess harness and four scenarios; the direct restoration function is 117 lines. Production restoration is a real feature that needs an owned seam, not incidental serialization.

This removes restoration feasibility as a blocker to transactional mid-round checkpoints. It does not prove that SQLite is necessary; that choice still follows from the accepted durability requirement.

## Evidence

Run:

```bash
uv run python prototypes/round_resume_checkpoint_prototype.py
```

The script checkpoints through JSON, starts a fresh Python process, restores new runtime objects, and compares the resumed result with uninterrupted execution.

| Boundary | Continuation | Result | Terminal |
| --- | --- | --- | --- |
| After initial setup | Scripted card decision | Identical prompt, Turn Record, and runtime state | No |
| After two Turns | Scripted card decision | Identical prompt, Turn Record, and runtime state | No |
| After two Turns | Retry exhaustion and RNG fallback | Identical prompt, fallback card, Turn Record, RNG state, and runtime state | No |
| After two Turns | Cache-backed correct Formal Guess | Identical prompt, verdict, Turn Record, evaluator state, and runtime state | Yes |

The structured checkpoints were 12–14 KB in these scenarios.

Static checks:

```bash
uv run ruff check prototypes/round_resume_checkpoint_prototype.py
uv run ty check prototypes/round_resume_checkpoint_prototype.py
```

Both passed. `aislop scan --changes` reported no errors and two complexity warnings for the deliberately single-file prototype: the 695-line file and 117-line restoration function.

## State required for continuation

- Ordered mainline, sidelines, hand, and remaining deck
- Failed guesses, Turn number, game-over state, and winner
- Secret rule definition and engine counters/settings
- Ordered completed Turn Records and next Turn index
- Scientist play history and `random.Random` state
- Provider-neutral prior usage needed for final accounting
- Rule compiler and validator caches when cache hits must survive resume
- Round settings and elapsed duration

The provider clients themselves do not need serialization. Resume reconstructs them from fixed Run settings. A crash during a Turn repeats that Turn and may repeat external calls, as already decided.

## Production seams exposed

The prototype deliberately reaches into private fields. Production code needs owned snapshot and restore operations instead:

- `Deck`: ordered-card snapshot and restoration
- `GameState`: structured snapshot and restoration, including a revealed hand
- `GameEngine`: continuation counters plus reconstruction from fixed settings
- `LLMScientist`: play-history and RNG restoration
- Round execution: start from an initialized checkpoint and a next-Turn index
- Usage and caches: provider-neutral persistence or explicit recomputation policy

These are bounded additions to existing modules introduced by commit `095a38a`; they do not require another round-execution rewrite.

## Limits

- The prototype uses a no-network model client. This matches the completed-Turn checkpoint boundary but does not test an interrupted provider call.
- It does not test SQLite transactions, crash injection, filesystem synchronization, or JSON export.
- It does not test compatibility across Python versions. Persisted RNG state implies that active checkpoints must require compatible runtime provenance.
- The Formal Guess scenario consumes a restored validator cache entry. It preserves the compiler cache but does not perform a fresh post-resume compilation.
- Wall-clock duration is reconstructed from accumulated elapsed time and excluded from exact equality.

## Design consequence

Proceed to the specification with per-run SQLite still in scope. Make exact snapshot/restore behavior the first tracer bullet and acceptance gate. Keep the reduced version-one scope: defer legacy-run import, generic invalidation tooling, multi-judge Shadow Verdict infrastructure, full compiler transcripts, and nested Provider Call telemetry.
