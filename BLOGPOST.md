# Are LLMs any good at the Science Game? Evaluating scientific reasoning using the card game Eleusis

Context:
 - Hugging Face Research Blogpost

Target audience:
- ML researchers interested in reasoning evaluation
- AI safety researchers (reasoning capabilities, calibration)
- Science communicators covering AI capabilities
- General technical audience curious about how LLMs "think"

Banner image:
**Figure**: LLM score vs token usage on a 2D scatter plot for each model.

## 1. Introduction

Large language models are increasingly being deployed as tools for scientific research—analyzing data, generating hypotheses, and even designing experiments. But how well do they actually embody the scientific method?

Most reasoning benchmarks test whether models can solve well-defined problems: given premises, derive a conclusion. The ARC challenge, for instance, evaluates inductive reasoning on visual patterns. These benchmarks capture important capabilities, but they miss something fundamental about how science actually works. Real scientific reasoning is not a single inference step. It's an iterative process of observation, hypothesis formation, experimentation, and refinement—often spanning many cycles before reaching a conclusion. It requires not just logical ability, but also *strategic* thinking: which experiment to run next, how much evidence is enough, when to commit to a theory versus when to keep exploring.

Beyond pure reasoning, effective science depends on psychological factors that are rarely evaluated: **calibration** (does my confidence match my actual accuracy?), **metacognition** (how certain am I about my uncertainty?), and resistance to **cognitive biases** like confirmation bias (seeking only evidence that supports my current hypothesis). A scientist who is brilliant at deduction but overconfident in weak theories will waste resources pursuing dead ends. One who is well-calibrated but overly cautious may never publish.

We wanted to test whether LLMs can exhibit these deeper aspects of scientific reasoning. To do this, we turned to an unlikely source: a 1950s card game called Eleusis.

Eleusis was designed by Robert Abbott explicitly to simulate the process of scientific discovery. In the game, one player invents a secret rule governing which cards can be played, and other players must deduce the rule through experimentation—playing cards and observing whether they are accepted or rejected. It's a microcosm of the scientific method: the rule is a hidden law of nature, each card play is an experiment, and the sequence of accepted and rejected cards is the accumulating evidence.

We built a benchmark around Eleusis to evaluate LLMs on this iterative, hypothesis-driven reasoning. Rather than testing knowledge retrieval or instruction-following, our benchmark asks: can models act like scientists? Can they observe evidence, form hypotheses, design informative experiments, and refine their theories? Can they calibrate their confidence appropriately and know when they've gathered enough evidence to commit to a conclusion?

These skills are fundamental not just to science, but to debugging code, diagnosing problems, and everyday reasoning under uncertainty.

**Figure**: Example of an Eleusis game sequence with the secret rule "alternating colors" (red, black, red, black...).

## 2. The Eleusis Benchmark

### The Original Game

In the original Eleusis card game, one player acts as the "dealer" (sometimes called "God" or "Nature") and secretly invents a rule determining which cards can be legally played. The other players don't know this rule—they must discover it through experimentation.

Players take turns playing cards from their hand onto a central "mainline." If a card satisfies the secret rule, it's accepted and added to the mainline. If it violates the rule, it's rejected and placed in a "sideline" below the mainline at that position. Over time, the pattern of accepted and rejected cards provides evidence about the hidden rule. At any point, a player can attempt to guess the rule; correctly identifying it ends the game. A specific scoring system rewards efficiency in discovering the rule while penalizing reckless guessing.

### Our Adaptation

We adapted Eleusis into a single-player benchmark focused purely on the scientific reasoning process. By removing multi-player dynamics, we isolate the core challenge: hypothesis formation and testing under uncertainty.

The game uses a standard 52-card deck with ranks 1–13 (Ace through King) and four suits. A secret rule—a deterministic function that takes the card being played and the current sequence of accepted cards (the "mainline")—determines whether each card is accepted or rejected. The player maintains a hand of 12 cards, drawing a replacement after each play.

On each turn, the player selects a card from their hand to play. If the card satisfies the secret rule, it joins the mainline; if rejected, it's placed in a sideline below the mainline at that position. At any point, the player may attempt to guess the rule.

The game lasts at most 30 turns, with scoring designed to reward efficiency while penalizing reckless guessing: `score = (30 - turns_used) - 2 × wrong_guesses`. A player who correctly identifies the rule on turn 10 with no wrong guesses scores 20 points; one who made 3 wrong guesses along the way scores only 14. Failing to identify the rule scores 0. This creates an interesting tension: guessing early yields more points if correct, but wrong guesses are costly. The optimal strategy requires accurately assessing one's own confidence—exactly the calibration we want to measure.

### Rule Library

We created a library of 26 hand-crafted rules spanning a range of types and complexity. Some rules involve simply card properties (e.g., "only red cards"), while others depend on the sequence of previously accepted cards (e.g., "card rank must be higher than previous card"). The rule might involve rank, suits, color or a combination thereof, and may include positional dependencies.

Example categories:

| Category | Examples |
|----------|----------|
| Static property | "Only red cards", "Only face cards (J, Q, K)" |
| Combined properties | "Only hearts with rank ≤7", "Only red face cards" |
| Sequential | "Rank must be higher than previous card" |
| Cyclic patterns | "Alternate between odd and even ranks", "Suits cycle: ♥→♠→♣→♦" |
| Complex conditionals | "Same suit as previous OR rank differs by exactly 2" |

Each rule is played 3 times with different random seeds (affecting the initial hand and deck order). This ensures every model is tested on the same deck sequences for a given seed, and captures variance in performance when the starting hand differs.

### What the LLM Must Do

On each turn, the model receives the complete game state: the mainline of accepted cards, the sidelines of rejected cards at each position, its current hand, and its history of reasoning from the 3 previous turns. It must output a structured response containing:

1. **Reasoning summary**: A brief explanation of its current thinking
2. **Card choice**: Which card to play from its hand
3. **Tentative rule**: Its current best hypothesis about the secret rule
4. **Confidence level**: A self-reported probability (0–10 scale, where 7 means "I estimate 70% chance my tentative rule is correct")
5. **Guess decision**: Whether to formally guess the rule this turn

This structure lets us analyze not just whether models succeed, but *how* they reason: Do they update hypotheses appropriately when evidence contradicts them? Do they explore strategically or play conservatively? Is their stated confidence calibrated to their actual accuracy?

**Figure**: Example turn showing the game state (mainline with sidelines) and the model's structured response. [Include both a well-reasoned turn and an example of overconfident incorrect reasoning]

## 3. Results

### 3.1 Overall Performance

Various level of performance across models but depends on model size and reasoning effort (token usage).

**Figure**:
- bar chart of Success rate (% of rounds solved)
- bar chart of Average score (efficiency)
- 2d scatter plot of average score vs output token count

### 3.2 Confidence and calibration

Models are asked to output their confidence level, with clear instructions on what it means (7 = 70% probability of being correct, etc).
Even when they don't guess, they have to report their tentative rule, when confidence >=5, we test whether they would have guessed it correctly.

**Figure**:
- Calibration curve: reported confidence (0-10) vs actual success rate with all models overlaid
- histogram of confidence levels when choosing to guess vs not guess
- 2D chart of average score vs average number of failed guesses per round. Check whether there is a correlation between reckless guessing and low score, or otherwise if some models are too cautious.

What to look for:
- Are models well calibrated? (do confidence levels match actual success rates?)
- Do some models guess too often or too rarely?
- Is there a tradeoff between cautiousness (few wrong guesses) and efficiency (high score)?

Can we measure for each model on average how many turns since they actually guessed the rule correctly, but choose not to guess it yet (missed opportunities) ?
Plot this vs average number of failed guesses ?

### 3.3 Rule analysis

**Figure**
- All rules with 1D scatter of the score obtained for each run of each model
- Heatmap of model vs rule success rate

Variance analysis ?
- Intra-rule variance: variation across seeds for same rule
- Inter-rule variance: variation across different rules

Report this for each model ?

### 3.4 Rule complexity

**Goal**: Understand how rule complexity affects model performance

Factor that influence complexity : average acceptance rate (some rules accept many cards, some few), code complexity of the rule implementation (cyclomatic complexity, AST node count)

Create a composite complexity measure combining these factors that maximally correlates with average score.

- Aggregated: `complexity = cyclomatic + k × node_count + y x abs(0.5 - acceptance_rate)` with k,y chosen to maximize correlation with relative score across all models/rules.

**Figure idea**: Scatter plot of success rate vs complexity.

Compare actual rule complexity and guessed rule complexity (to judge whether models are overfitting/underfitting).

Some semantic aspects of complexity not captured by code metrics 
(e.g., "only faces" is harder than "only 3, 6 and 7" even though both have similar code complexity).


## 4. Deeper analysis and anecdotes

### 4.1 Learning Curves

**Question**: How do models improve within a round?

**Analysis**:
- Track confidence over turn number
- Track acceptance rate over time (should decrease as obvious cards are exhausted)

**Figure idea**: Line plot of avg confidence by turn, colored by success/failure

### 4.2 Failure Modes

**Question**: When models fail, why?

**Taxonomy** (needs analysis):
1. **Premature guessing**: High confidence, wrong rule, insufficient evidence
2. **Hypothesis fixation**: Stuck on wrong rule despite contradictory evidence
3. **Overfitting**: Rule matches observations but is more specific than actual rule
4. **Underfitting**: Rule is too simple (e.g., "black cards" when rule is "black even cards")
5. **Position blindness**: Fails on rules depending on position in mainline

**Figure idea**: Stacked bar of failure modes by model

**Qualitative example**: Show a failed round's turn-by-turn reasoning

### 4.3 Misc observations

Symmetric rules
"Only spades" vs "Only non-spades" should be equally difficult, but are they?

Confirmation bias
To they play cards that confirm their hypothesis more often than disconfirming ones?

## 5. Conclusion 

1. **LLMs can do inductive reasoning**—but with significant variation
2. **Complexity matters**—simple rules are easy, complex rules are hard (not surprising, but now quantified)
3. **Calibration is imperfect**—models don't always know what they don't know
4. **Reasoning traces are valuable**—the turn-by-turn data reveals how models think


Limitations:
**Rule library scope**: 26 hand-crafted rules may not cover all reasoning types
**Statistical caveats**: 3 seeds per rule may not capture full variance
**Prompt sensitivity**: Different prompts might yield different results
**No human baseline**: Hard to compare to human performance on same rules
**Cost/API differences**: Models have different pricing, not directly comparable

- What's next: more models, more rules, human comparisons
- Call to action: benchmark is open source, try it yourself

## Appendix : Detailed methods

### Models 
API calls
Reasoning effort
temperature
max tokens
retry

### Rule checking using python code and the rule compiler LLM
Rules created by hand
Compiled into python functions using an LLM, manually verified
When LLM outputs a guessed rule, compile and test against all played cards to check correctness
Test bby simulation rather than natural language matching, because of semantic equivalence, and unknowable variations in wording.
(for instance "same color as previous card" vs "red cards only" once the first card is played)

### Prompt structure
Rules of the game, no reference to Eleusis name to avoid leakage, details on scoring system
Format of the response (JSON with specific fields)
Detailed state of the game : mainline, sideline, hand, previous turns and their reasoning, previous failed guesses
JSON output

## References
- Abbott, R. (1963). "Eleusis" (original game rules)
- Cognitive science papers using Eleusis
- Recent LLM reasoning benchmarks (ARC, BIG-Bench, etc.)
- Calibration literature (Guo et al., 2017)
