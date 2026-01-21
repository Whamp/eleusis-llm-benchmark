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

LLM are more and more used for scientific research.
Typical benchmark like ARC test inductive reasoning but science is more than that.
It requires observation, hypothesis formation, experimentation, refinement and iteration. 
This is an iterative agentic / autonomous behavior of scientific method.
Not tested by typical benchmarks.
Depends not only on pure reasoning but also some psychology aspect like calibration, metacognition, cognitive biases, strategic experimentation, risk taking.

Eleusis is a card game that simulates this process. 
We built a benchmark around Eleusis to test whether LLMs can perform inductive reasoning by playing the game.
Rather than testing knowledge retrieval or following instructions, Eleusis tests whether models can act like scientists: observing evidence, forming hypotheses, testing predictions, and refining theories.
It requires inductive reasoning, building a mental model of hidden rules from limited data, and strategic experimentation. It requires research taste, calculated risk taking, avoiding overfitting, and knowing when to trust one's conclusions, metacognition (how confident am I in this hypothesis?), calibration (do I know what I don't know?), and scientific humility (knowing when to withhold judgment), not falling into cognitive biases like confirmation bias.
This is fundamental to science, debugging, and everyday problem-solving. 

**Image** What is Eleusis : picture with an example of a sequence with secret rule "alternating colors".

## 2. The Eleusis benchmark

Original game : one player is the dealer (sometimes called God or Mother nature) who invents a secret rule for which cards can be played. Other players try to figure out the rule by playing cards and observing which are accepted or rejected.
Players take turn in playing cards from their hand to a mainline pile. If the card follows the secret rule, it is accepted and added to the mainline; otherwise, it is rejected and placed in a sideline pile below. Players can also guess the secret rule at any time. 
The game ends when a player correctly identifies the rule or when a maximum number of turns is reached.

We turned this into a solo benchmark, focus on scientific method and less on game strategizing, limiting the number of free parameters.

### How does this work ?

52-card deck (ranks 1-13, 4 suits)
Secret rule: a function `(card, mainline) → accept/reject`
Player receive a hand of 12 random cards.
Each turn:
- selects a card from their hand to play
- receives feedback: accepted (added to mainline) or rejected (added to sideline)
  - may optionally try to guess the rule : if correct, game ends, otherwise penalty applied.
Scoring system : 30 turns to guess the rule, receive (30-turns_used) - penalty × wrong_guesses with 2 points penalty per wrong guess.

The penalty means the player must carefully assess their confidence before trying to guess the rule.
- Rewards efficiency (fewer turns = higher score)
- Penalizes reckless guessing

We created a library of 26 hand-crafted rules of varying complexity, spanning simple set membership (e.g., "only red cards") to complex conditional patterns (e.g., "suits must appear in pairs").

**Table idea**: Example rules from each category with brief description.


Each rule played 3 times with a different random seed (hand + card order) to capture variance.

- What the LLM must do:
  - Reason about the current state of the game, knowing their history of play, accepted/rejected cards, wrong guesses.
  - Choose a card to play
  - Maintain a "tentative rule" hypothesis
  - Report confidence level (0-10)
  - Decide whether to guess or not

**Figure idea**: Example of a JSON showing a mainline/sideline and turn's response: reasoing summary, tentative rule, confidence level, played card
Show two examples : a good round and a bad round (with overly confident wrong guesses and complex rule)

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
