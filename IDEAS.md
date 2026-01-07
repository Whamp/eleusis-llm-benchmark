
# IDEAS FOR IMPROVEMENTS

## Future Analysis Ideas (LLM-as-judge, embeddings, manual labeling)

These extend beyond the quantitative metrics in `scripts/analyze_results.py`.

### 1. Reasoning Intent Classification (LLM-as-judge)

**Goal**: Understand whether models are playing strategically or just trying to satisfy the rule.

Classify each turn's `reasoning_summary` into:
- **Confirmation**: Trying to get accepted (support current hypothesis)
- **Falsification**: Deliberately testing if a card will be rejected
- **Exploration**: Testing to discriminate between multiple hypotheses

**Key questions**:
- Do models that falsify more often succeed more often?
- How does intent distribution change over turns (early exploration → late confirmation)?

### 2. Hypothesis Evolution Analysis (Embeddings)

**Goal**: Track how `tentative_rule` evolves toward the actual rule over turns.

Use sentence embeddings to compute similarity between each turn's `tentative_rule` and the actual `rule_description`, then plot the trajectory.

**Key questions**:
- Do models show "eureka moments" (sudden jump in similarity) or gradual convergence?
- Are there cases of "hypothesis regression" (getting closer then farther)?

### 3. Failure Mode Taxonomy (Manual + LLM)

**Goal**: Categorize why models fail.

**Categories**:
- **Never found**: `tentative_rule` never approximated actual rule
- **Found but didn't guess**: Had correct hypothesis but waited too long
- **Overconfident wrong**: Multiple high-confidence incorrect guesses
- **Complexity overwhelm**: Hypothesis kept getting more complex without settling

### 4. Strategic Card Selection Quality (Simulation)

**Goal**: Evaluate whether the played card was informative given the hand and hypothesis.

For each turn, compute "information gain" - how much the played card narrowed down the hypothesis space.

### Priority Order

1. **Failure Mode Taxonomy** - Most actionable for understanding model weaknesses
2. **Hypothesis Evolution** - Visually compelling, shows reasoning process
3. **Reasoning Intent** - Reveals strategic vs non-strategic behavior
4. **Card Selection Quality** - Most technically complex, do last
