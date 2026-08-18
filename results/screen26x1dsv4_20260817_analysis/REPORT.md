# DeepSeek V4 Flash 0731: screen_26x1

## Result

The run completed all 26 scheduled Rounds. DeepSeek V4 Flash 0731 won 25 Rounds with a correct Formal Guess and reached the 30-turn limit on one Round:

- **Success:** 25/26 (96.2%)
- **Turn-limit Round:** `paired_ranks_distinct`
- **Authoritative Turns:** 282
- **Total score:** 389
- **Average score:** 15.0
- **Wrong Formal Guesses:** 67
- **Fallback Decisions:** 9

The 25 successful Rounds all ended with `correct_formal_guess`. The turn-limit Round did not produce a correct Formal Guess by turn 30.

## Run identity and condition

- **Benchmark model:** `deepseek-v4-flash-0731`
- **Endpoint:** OpenRouter, pinned to Cloudflare with `deepseek/deepseek-v4-flash-0731:cloudflare`
- **Reasoning:** provider-native streamed reasoning deltas, persisted as `reasoning_text`
- **Temperature:** 0.7
- **LLM seed:** 42
- **Batch round index:** 1 for every Rule
- **Maximum Turns:** 30
- **Rules:** all 26 rules from `rules.json`, one Round per Rule
- **Rule compiler:** OpenRouter-pinned Novita `gpt-oss-120b` primary with configured fallbacks
- **Shadow judge:** 100 simulations × 40 turns, simulation seed 42
- **Source revision:** `acf629714aef8c5ebe847e84763d38c2f609a31f`

The run used the model-specific 32,768-token allowance added for DeepSeek's long reasoning. The Round Record's stored `settings.llm.max_tokens` still reports the global 16,384-token value. The source fingerprint records the exact `models.yaml` used, but the exported setting is misleading and needs a follow-up fix. Completed Round Records were not edited after the run.

## Reasoning and usage

The model generated unusually long native reasoning traces:

- **Output tokens:** 7,748,804
- **Reasoning tokens:** 7,683,179
- **Answer tokens:** 65,625
- **Reasoning share:** 99.2% of output tokens
- **Aggregate provider-call time:** 81,916 seconds
- **Max-token retries:** 81

The 32K allowance prevented most truncations, but harder Rules still exceeded it and retried. These retries were recorded as `max_token_reached`; they did not create fabricated cards. The final two Rounds required 26 and 30 Turns respectively, with repeated truncation retries.

## Shadow Guesses

Offline evaluation stored **104 Shadow Verdict sidecars** in the authoritative SQLite databases and regenerated every portable `results.json` export.

- **Correct Shadow Verdicts:** 15/104 (14.4%)
- **Successful Rounds with a correct Shadow Guess before the Formal Guess:** 10/25 (40.0%)
- **Median lead before the winning Formal Guess:** 1 Turn

Shadow correctness is deliberately separate from Formal Guess success. A Shadow Guess records a counterfactual verdict and does not affect gameplay or score.

## Rule difficulty

Lowest floored scores came from the Rules that require tracking relationships across multiple cards:

- Paired ranks distinct: 0, turn limit
- Paired suits alternating: 0, won on turn 26
- Face cards impose suit: 3
- Face cards red / number cards black: 1

The analysis projection reports the lowest complexity-quartile success rate as 71.4%, compared with 100% for the lowest quartile. The aggregate complexity-success correlation was -0.315.

## Analysis caveat

The authoritative SQLite Round Records contain 282 Turns. The generated legacy analysis tables report 259 Turns. The discrepancy is in the analysis projection, not the stored Round Records or exported structured records. Headline counts above use the authoritative SQLite data; projection-derived charts and tables are retained as generated artifacts and should not be used for exact Turn totals until the loader discrepancy is resolved.

## Artifacts

- 26 worker directories under `results/solo_evaluation_*screen26x1dsv4_w*/`
- Each worker contains `benchmark_run.sqlite3` and regenerated `results.json`
- Aggregate analysis: this directory
- Summary: `summary.txt`
- Machine-readable metrics: `basic_metrics.csv` and the generated JSON files
- Charts: generated PNG files in this directory

All 26 worker databases pass SQLite integrity checks. No benchmark worker processes remained after completion.
