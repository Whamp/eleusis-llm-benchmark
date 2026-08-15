# Eleusis LLM Benchmark

This benchmark tests inductive reasoning through Eleusis: a secret rule accepts or rejects
played cards, and the benchmark model tries to infer that rule from observed outcomes.

## Working in This Repository

- Use `uv`, including for dependency changes. Install with `uv sync`.
- Target `Whamp/eleusis-llm-benchmark` for pushes, issues, and pull requests. Treat
  `scienceetonnante/eleusis-llm-benchmark` as fetch-only upstream.
- Run tests with `uv run pytest` and lint with `uv run ruff check`.
- Smoke-test a model before a full run:
  `uv run python scripts/evaluate_single.py --config config.smoke.yaml --model <key> --tag smoke`.
- Run the default full benchmark with
  `uv run python scripts/evaluate_single.py --model <key>`.
- Use `--suite <name>` for a named case set from `suites.yaml`; a suite selects rule names and
  `batch_round_index` values.
- Read `README.md` when running, parallelizing, resuming, monitoring, or analyzing benchmarks.

## Domain Vocabulary and Flow

- **Rule compiler:** the model configured under `rule_compiler` in the run config. It converts
  natural-language rules—including benchmark-model guesses—into executable Python.
- **Benchmark model:** the model under evaluation, selected by `--model <key>` from `models.yaml`.
  `src/eleusis/player.py` implements its `LLMScientist` player.
- **Secret rule:** the compiled `Rule` that decides whether a card joins the mainline or a
  sideline.
- **Shadow guess:** a tentative rule evaluated without changing gameplay or score. The default
  `shadow_mode: offline` records it for later processing by `scripts/evaluate_shadows.py`.

`scripts/evaluate_single.py` resolves the run config, benchmark model, and optional suite, writes
checkpoints and results, then calls `src/eleusis/runner.py` for each round. The runner coordinates
`LLMScientist`, the game engine, and rule validation. Provider adapters live under
`src/eleusis/llm/`; post-hoc reporting lives under `src/eleusis/analysis/`.

## Behavioral Invariants and Risk Boundaries

- `src/eleusis/game/engine.py` executes generated rule bodies with `exec()` and an allowlisted
  namespace. Treat the rule-compilation path and `safe_globals` as security-sensitive; this is
  restricted execution, not a security boundary for hostile code.
- Rule guesses are compiled and compared with finite, seeded simulations. Comparison is neither
  string matching nor a proof of logical equivalence.
- The game uses two standard decks (104 cards). A normal card play draws a replacement, keeping
  the configured hand size constant while cards remain in the deck.
- A seeded round uses
  `(base_seed + low_32_bits(md5(rule_code)) + batch_round_index) & 0xffffffff`.
  The same rule and batch index reproduce a shuffle; different batch indices vary it.
- Each round creates fresh game state. Reusing a secret rule never reuses the preceding round’s
  deck, hand, mainline, or sidelines.
- Keep provider credentials in the gitignored `.env`; use `.env.example` as the key template.

## Agent skills

### Issue tracker

Issues live in GitHub Issues at `Whamp/eleusis-llm-benchmark`. See
`docs/agents/issue-tracker.md`.

### Triage labels

Use the five canonical triage labels without aliases. See
`docs/agents/triage-labels.md`.

### Domain docs

Use the single-context layout: root `CONTEXT.md` plus `docs/adr/`. See
`docs/agents/domain.md`.
