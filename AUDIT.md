# Scientific Audit: Eleusis LLM Benchmark

**Date**: 2026-01-19
**Scope**: Codebase review for scientific soundness prior to publication as research blog post
**Goal**: Benchmark LLM abilities in inductive reasoning, hypothesis testing, and calibration using a card game derived from Eleusis

---

## Executive Summary

The Eleusis benchmark is a well-structured system for evaluating LLM reasoning capabilities. The core game mechanics are sound, and the data collection is comprehensive. However, several design choices have unexamined implications for cross-model comparison and scientific validity.

**Key strengths:**
- Rich turn-by-turn logging enables detailed analysis
- Calibration analysis (confidence vs correctness) is particularly valuable
- Checkpointing enables reproducible evaluation runs
- Rule library with complexity metrics allows difficulty analysis

**Key concerns:**
- Rule comparison uses probabilistic simulation (potential false negatives)
- Rule compilation errors conflated with semantic errors
- LLM seed parameter ignored for most providers
- Sample size adequate for large effects only (~±10% CI on success rate)

---

## 1. Critical Validity Issues

### 1.1 Rule Comparison is Probabilistic, Not Deterministic

**Severity: HIGH**

The `compare_rules()` function (`src/eleusis/game/validator.py:108-139`) uses simulation-based equivalence testing:
- Only 2 simulations × 10 turns × 52 cards = ~1,040 comparisons
- Card sequence within each simulation chosen via `random.choice()` (line 212) using **unseeded global RNG**
- The same guess can theoretically be judged correct or incorrect depending on random path taken

**Example vulnerability**: A rule like "ranks must strictly increase, Ace may follow King" could be judged differently depending on which random card sequences the simulation explores.

**Recommendations:**
1. Seed the RNG in simulation loop for reproducibility
2. Increase simulation count for mainline-dependent rules
3. Consider exhaustive comparison for rules with small state spaces

---

### 1.2 Rule Compilation Conflates Semantic Error with Translation Error

**Severity: HIGH**

When a player's natural language guess is converted to code via `rule_compiler_client.convert_rule_to_code()`, compilation failures are indistinguishable from semantic incorrectness.

The player is penalized equally for:
1. Their rule being semantically wrong (intended measurement)
2. The compiler misunderstanding their description (confound)
3. The compiler generating syntactically invalid code (confound)

**Evidence**: `validator.py:151-158` - compilation failures return `False` with generic error message, counted as wrong guess.

**Recommendations:**
1. Track compilation failures separately from semantic failures
2. Add `guess_failure_reason` field: `compilation_error`, `syntax_error`, `semantic_mismatch`
3. Consider human review of failed guesses to estimate compiler error rate
4. Report percentage of guesses that failed to compile vs semantically wrong

---

### 1.3 LLM Seed Parameter Is Ignored for Most Providers

**Severity: MEDIUM**

Config specifies `llm_seed: 42`, but implementation shows:
- Anthropic, OpenAI, and Google clients store but don't pass seed to API
- Only HuggingFace and XAI clients actually use seed parameter

**Impact**: Results for Claude, GPT, and Gemini models are not deterministically reproducible even with identical configuration.

**Recommendations:**
1. Either implement seed passing for all providers, or
2. Remove seed from config and document that LLM outputs are non-deterministic
3. Note in methodology that "seed" provides partial reproducibility only

---

## 2. Fairness Concerns Across Models

### 2.1 Reasoning Token Asymmetry

Models with extended thinking (Claude Opus, o-series, DeepSeek-R1) produce many reasoning tokens that are counted in usage statistics.

**Example from results**: Claude Opus 4.5 used 13,061 reasoning tokens vs 845 answer tokens in one round.

**Impact**: Token efficiency comparisons may unfairly penalize reasoning models.

**Recommendations:**
1. Report metrics both with and without reasoning tokens
2. Separate `reasoning_tokens` from `answer_tokens` in efficiency analysis
3. Consider cost-normalized metrics (output per dollar) alongside token counts

---

### 2.2 Prompt Length Sensitivity

The game prompt includes substantial context:
- Full game rules explanation (~500 tokens)
- Last 10 play history with accept/reject status
- All failed guesses with explicit "THESE WERE INCORRECT" marker
- Scoring formula and current score projection

Different models may have varying sensitivity to prompt structure and length.

**Recommendations:**
1. Track prompt token counts per model
2. Check for correlation between prompt length and performance
3. Consider prompt ablation study (with/without failed guess history)

---

### 2.3 Single Rule Compiler for All Models

All models' natural language guesses are compiled by the same model (`gpt-oss-120b`).

**Tradeoffs:**
- **Advantage**: Consistent interpretation standard across all models
- **Disadvantage**: Models that phrase rules in ways gpt-oss-120b doesn't understand well are systematically penalized

**Recommendations:**
1. Document this as a methodological choice
2. Consider sensitivity analysis: have each model compile their own guesses
3. Report compiler success rate by source model

---

### 2.4 History Truncation

Only the last 10 plays are shown in the prompt, but early patterns may be crucial for rule discovery.

**Impact**: Could disadvantage models that benefit from full trajectory analysis.

**Recommendation**: Document this design choice; consider variable window or full history option.

---

## 3. Statistical and Methodological Issues

### 3.1 Sample Size and Statistical Power

With 20-25 rules × 5 rounds = 100-125 observations per model:

**For success rate (binomial):**
- If true success rate is 50%, SE = √(0.5×0.5/100) = 0.05
- 95% CI width ≈ ±9.8%
- Detectable difference: ~20 percentage points with 80% power

**For average score (continuous):**
- Detectable effect size: ~0.3-0.4 Cohen's d
- Adequate for large differences, not subtle ones

**Recommendations:**
1. Report confidence intervals on all metrics, not just point estimates
2. Be explicit about detectable effect sizes in methodology
3. Avoid over-interpreting small differences between models

---

### 3.2 Non-Independence of Rounds with Same Rule

Design: 5 rounds per rule to "check for standard deviation"

**Issue**: With `seed: 42`, the same rule produces deterministic initial deck/hand. The 5 rounds are not independent samples - they share:
- Same rule interpretation difficulty
- Same initial game state (modulo LLM non-determinism)

**What this actually measures**: Variance from LLM non-determinism, not rule difficulty variance.

**Recommendations:**
1. Use different seeds per round within a rule (measures rule × LLM interaction)
2. Or acknowledge this explicitly: "variance reflects LLM stochasticity, not rule variance"
3. Consider whether 5 identical-setup rounds is the right design

---

### 3.3 Rule Selection Bias

The `rules.txt` contains 20 human-curated rules that are:
- All interpretable by humans
- All have clear Python implementations
- Represent patterns humans find "interesting"

**Impact**: The benchmark measures "ability to discover rules that humans find interesting and can articulate" not "ability to discover any logical pattern."

**Recommendations:**
1. Document rule selection criteria explicitly
2. Consider adding adversarial/edge-case rules
3. Note that generalization beyond this rule set is not established

---

### 3.4 Deck Composition

Using a double deck (104 cards) with duplicates:
- Some cards appear twice in the same hand (observed in results: "4♣, 4♣")
- May affect rule discovery for rank/suit-based patterns

**Recommendation**: Document this choice; consider whether single deck would be cleaner.

---

## 4. Missing Features and Recommended Additions

### 4.1 Baseline Comparisons

No baselines currently included. Recommended additions:

1. **Random baseline**: Random card selection + never guess
   - Establishes floor performance
   - Quantifies "how much does reasoning help?"

2. **Heuristic baseline**: Simple strategy (e.g., maximize information gain)
   - Establishes what's achievable without LLM

3. **Human performance estimate**: Even rough calibration would help
   - "Expert human solves X% of these rules in Y turns"

---

### 4.2 Error Analysis Categories

Currently tracked: `failed_guesses` count

**Missing:**
- Why guesses failed (semantic vs compilation error)
- Pattern of incorrect hypotheses (what did models think the rule was?)
- Card play strategy quality (were rejections used informatively?)
- Near-miss analysis (guesses that were "almost right")

**Recommendations:**
1. Add `guess_failure_reason` field with categories
2. Log the compiled code for failed guesses (already done)
3. Consider semantic similarity metric for "how close" wrong guesses were

---

### 4.3 Shadow Evaluation Confound

The codebase includes "shadow evaluation" (`runner.py:291-313`):
- If player has confidence ≥ 5 but doesn't officially guess
- System secretly evaluates their tentative rule
- Recorded for analysis but doesn't affect gameplay

**Issue**: Creates asymmetric information - analyst knows more about model reasoning than model knows about evaluation. Could bias calibration analysis.

**Recommendations:**
1. Either disable shadow evaluation for clean analysis, or
2. Clearly separate shadow results from official guess results
3. Document this feature explicitly in methodology

---

### 4.4 Rule Difficulty Normalization

Different rules have different intrinsic difficulties. Current analysis shows score by complexity metrics, but:
- No normalization for "baseline difficulty"
- A model could look good by succeeding on easy rules, failing hard ones
- Aggregate metrics hide rule-level variation

**Recommendations:**
1. Report per-rule × per-model matrix (heatmap)
2. Consider rule-level random effects in statistical analysis
3. Compute rule difficulty from cross-model average, then normalize

---

## 5. Reproducibility Assessment

### 5.1 What IS Reproducible

| Component | Status | Notes |
|-----------|--------|-------|
| Rule library | ✓ | `rules.json` with full metadata |
| Deck shuffle | ✓ | Seeded with rule hash + base seed |
| Config parameters | ✓ | Saved in results.json |
| Turn-by-turn logs | ✓ | Complete action/response history |
| Checkpointing | ✓ | Can resume interrupted evaluations |

### 5.2 What IS NOT Reproducible

| Component | Status | Notes |
|-----------|--------|-------|
| LLM outputs | ✗ | Seed not passed to Anthropic/OpenAI/Google |
| Rule comparison | ✗ | Unseeded simulation RNG |
| Token counts | ~ | Provider-specific estimation methods |

### 5.3 Recommendations for Reproducibility

1. Publish full results.json files (not just aggregates)
2. Document exact model versions used (not just model names)
3. Include rules.json in publication
4. Note that exact replication requires same API versions

---

## 6. Recommendations Summary

### Before Running Final Evaluation

| Priority | Action |
|----------|--------|
| HIGH | Seed the simulation RNG in `compare_rules()` |
| HIGH | Track rule compilation failures separately |
| MEDIUM | Document that LLM seed is advisory only |
| MEDIUM | Consider increasing simulation count for equivalence testing |

### For Analysis and Reporting

| Priority | Action |
|----------|--------|
| HIGH | Report confidence intervals on all metrics |
| HIGH | Include random baseline for context |
| HIGH | Show per-rule × per-model heatmap |
| MEDIUM | Separate reasoning tokens in efficiency metrics |
| MEDIUM | Document rule selection criteria explicitly |

### For Blog Post Framing

| Recommendation |
|----------------|
| Frame as "exploratory benchmark" not definitive ranking |
| Acknowledge rule compiler as potential confound |
| Be explicit about detectable effect sizes (~20% difference in success rate) |
| Emphasize calibration analysis as key contribution |
| Note limitations on generalization beyond tested rule types |

---

## 7. Conclusion

The Eleusis benchmark is a creative and well-implemented approach to evaluating LLM reasoning. The core mechanics are sound, and the comprehensive logging enables rich analysis. The calibration analysis (confidence vs actual correctness) is particularly valuable and should be emphasized.

The main threats to validity are:
1. The rule compilation step introduces an uncontrolled confound
2. The probabilistic rule comparison could produce false negatives
3. Sample size limits detection to large effects only

With the recommended fixes and appropriate framing, this benchmark can provide meaningful insights into LLM reasoning capabilities, particularly around:
- Inductive reasoning from examples
- Hypothesis testing and refinement
- Metacognitive calibration (knowing what you know)

The results should be framed as exploratory findings on a specific task rather than definitive rankings of general reasoning ability.