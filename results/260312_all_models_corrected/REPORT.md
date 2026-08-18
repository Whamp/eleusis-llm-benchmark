# All-models comparison plus DeepSeek V4 Flash 0731 (screen_26x1)

## Result

`report.html` and the generated charts in this folder now include **14 models**: the original 13 historical 78-round runs plus DeepSeek V4 Flash 0731 from the refactored `screen_26x1` suite.

On average floored score under the current analysis loader:

| Rank | Model | Rounds | Success | Avg floored score |
| --- | --- | --- | --- | --- |
| 1 | Deepseek V4 Flash 0731 | 26 | 88.5% | 17.00 |
| 2 | Claude Opus 4.5 | 78 | 83.3% | 16.41 |
| 3 | Kimi K2 | 78 | 76.9% | 15.58 |

DeepSeek's 26-round headline matches its standalone analysis: 25/26 Formal Guess wins, one turn-limit Round (`paired_ranks_distinct`), average score 15.0, average floored score 17.0. Full run notes: `results/screen26x1dsv4_20260817_analysis/REPORT.md`.

## This is not a like-for-like ranking

The DeepSeek row is here so it can be read against the old leaderboard. It is **not** the same experimental condition:

- **Rounds.** Historical models played 26 rules × 3 shuffles (`full_26x3`, 78 Rounds). DeepSeek played 26 rules × 1 shuffle (`screen_26x1`, `batch_round_index` 1 for every Rule). Intra-rule variance is undefined for DeepSeek.
- **Stack.** DeepSeek used the refactored SQLite Round Record path, OpenRouter-pinned compiler waterfall, and a 32,768-token player allowance. The historical runs used the pre-refactor JSON results format and a different compiler setup.
- **Tokens.** DeepSeek averaged about 23.6k output tokens per Turn, almost all native reasoning. That is far above every historical model in this folder.
- **Analysis projection.** DeepSeek's authoritative SQLite stores 282 Turns; the generated tables here report 259 Turns, the same projection discrepancy noted in the standalone report.

Do not treat DeepSeek's #1 floored-score rank as a 78-round result.

## Historical numbers also moved

Re-running `scripts/analyze_results.py` with the current loader recomputes derived values from Turn facts. Historical models keep the same 78 Rounds, but some derived columns differ from the original March snapshot. Examples:

- Claude Opus 4.5 average floored score: 16.97 in the original snapshot, 16.41 here
- GPT 5.2 High total Turns: 1,195 originally, 1,264 here

`results_report.html` is the original editorial 13-model × 78-round snapshot and was not regenerated. Use it for the numbers that were published with the corrected Qwen run. Use `report.html` for the mixed-protocol comparison that includes DeepSeek.

## Artifacts

- Comparison charts, `summary.txt`, `basic_metrics.csv`, and `report.html` in this folder
- Historical runs: relative links into `results/260121_78_rounds/` and `results/solo_evaluation_20260312_qwen3_5_27b_gptq_int4_corrected/`
- DeepSeek workers: relative links into `results/solo_evaluation_*screen26x1dsv4_w*/`
- Standalone DeepSeek report: `results/screen26x1dsv4_20260817_analysis/`
