# Benchmarking in-context inductive reasoning of LLMs in the card game Eleusis

*Note: This is a preliminary version of the benchmark description. For now the benchmark has only been run at small scale on a few models as a proof of concept.*

Large language models exhibit striking in-context learning abilities. While it started with few-shot learning of linguistic tasks, they have now succeeded in more complex domains, such math competition like IMO, or even assisting mathematicians by proving useful intermediate lemmas in research-level mathematics. Despite those successes, there is still a vivid debate about whether LLMs can truly create internal world models, that is to infer underlying rules of the world from limited data, and use them for prediction, simulation, and decision making.

The card game *Eleusis* is a rule-discovery game that has often been considered as an analogy of the scientific method. In Eleusis, one player, called *the Rule-maker*, chooses a secret rule that defines which cards may be played in sequence. The other players, called *the Scientists*, play cards from their hand in a shared sequence, and try to uncover the secret rule. For each card played, the Rule-maker simply states whether the card is consistent with the rule or not. A Scientist who feels confident enough can propose an explicit formulation of the secret rule to try to end the game.

Playing Eleusis requires inductive reasoning and adopting a trial and error mindset: the Scientists must (1) Form hypotheses about the secret rule based on the limited evidence provided by the Rule-maker’s feedback, (2) test these hypotheses by choosing which cards to play next, (3) revise their hypotheses in light of new evidence, and (4) assess their own uncertainty (meta-reasoning) to decide when to propose a rule.

Although Eleusis has been discussed in the context of science education , and sometimes considered as a testbed for symbolic AI systems in the 80s, to the best of our knowledge, it has never been used as a benchmark for large language models. Compared to traditional LLM benchmarks, Eleusis is inherently multi-turn, open-ended, and but still comes with a precise ground truth. Its concept space is symbolic but natural and grounded in everyday language, dealing with card ranks, colors, suits, and simple relations between them.

That makes Eleusis a compact, controlled setting to study in-context inductive reasoning in large language models.

Several benchmarks probe inductive reasoning in LLMs, but Eleusis occupies a distinct niche. ARC (Abstraction and Reasoning Corpus) and Raven's Progressive Matrices test single-shot visual pattern completion from static examples—the model sees the evidence and makes one prediction. List function and string transformation tasks are closer in spirit but typically provide examples upfront rather than letting the agent gather them. Eleusis differs in three key ways: (1) it is active—the agent chooses which hypotheses to test by selecting cards, rather than passively receiving examples; (2) it is sequential—hypotheses must be revised over many turns as evidence accumulates; and (3) it requires calibrated confidence—the agent must decide when to commit to a guess, balancing the risk of premature commitment against the cost of continued play. This combination of active hypothesis testing, iterative refinement, and meta-cognitive decision-making mirrors the structure of scientific inquiry more closely than existing benchmarks.




## Rules of Eleusis

This is a short description of the rules we have been using for the benchmark. They are a slightly simplified version of vanilla Eleusis (although many variants exists for the card game).

**Setup.** One player (the *Rule-maker*) secretly chooses a rule governing which cards may legally extend a sequence. Rules may reference card properties (rank, suit, color, face/non-face) and the visible sequence, but not hidden information. Example rules: "alternate red and black," "each card must be higher than the previous," "only prime ranks after a face card."

The Rule-maker plays one legal card to start the *main line*. Each Scientist receives a 12-card hand.

**Turns.** On your turn, either:

- **Play a card.** The Rule-maker announces *accepted* (card joins the main line; hand shrinks by 1) or *rejected* (card is placed below the main line; draw 2 cards as penalty).
- **Declare no-play.** Claim no card in your hand is legal. If wrong, the Rule-maker plays a valid card from your hand and you draw 3. If correct, discard your hand and receive a new hand with 3 fewer cards.

**Guessing.** After any *successful* action (accepted card or valid no-play), you may propose a rule. If the Rule-maker judges it equivalent to the secret rule, the game ends and you receive a −6 point bonus. If wrong, draw 2 cards.

**Scoring.** The game ends when someone guesses correctly or after 40 turns. Your score is the number of cards remaining in your hand (lower is better), minus 6 if you guessed the rule.

The following image shows an example game state. The secret rule is *"rank of the card must be strictly greater than the previous card; any card might be played after a King"* The main line is on top and the rejected cards are below. We can see for instance that the 3♠️ was rejected because it is not strictly greater than the previous card (4♥️), same for the 8♥️ and 2♥️ after the 8♣️.
![Eleusis game state](https://upload.wikimedia.org/wikipedia/commons/b/be/Eleusis_card_game.jpg)




## Benchmark
In our benchmark, we task LLMs with playing the role of Scientists in Eleusis games with various secret rules. We used another LLM as the game master for various tasks such as translating natural language rules into verifiable Python functions.

### Methodology
#### Rule creation

We first created a list of rules, aiming for rules that are not too complicated nor too easy (in the real game, there is a scoring mechanism for the rule maker that incentives them to create rules that are a sweet spot). We instructed the game master to produce a Python function that could be used for verifying card plays and rule guesses.

#### Gameplay
We created a prompt template to instruct the models playing the Scientists. Each game starts with a system prompt reminding the model of its role and the rules of the game. The prompt then exposes the current state of the game:
- the state of the mainline (accepted cards) and the rejected cards at each position;
- the current hand of the model;
- a reminder of their previous plays and the Rule-maker's feedback;
- the list of all the previous failed attempts at guessing the rule (by any of the players, since rule guessing is public and a failure provides information to all players);

The model is then asked to reason about the current situation and choose an action (a card to play or declare no-play.) Answers are expected in a structured JSON format

```json
{
    "reasoning": "Your analysis of the pattern and why you're playing this card/no-play",
    "action": "5♥" or "no_play",
    "tentative_rule": "Your current best guess about the rule (always provide this)",
    "confidence_level": 0-10 (your confidence in the tentative_rule, 0=lowest, you have no clue, 10=maximum, you are 100% sure),
    "guess_rule_if_accepted": true or false (whether to officially guess if accepted)
}
```

#### Adjudication
The action of the model is then executed in the game environment, and feedback from the Rule-maker is provided (we use the ground truth Python function describing the rule to provide accurate feedback). 

If the action is successful (card accepted or succesful no-play) and the model has chosen to guess the rule, the guess is evaluated for equivalence with the ground truth rule. For this, we first prompt the game master LLM to convert the proposed rule in natural language into a Python function. Then, we test the equivalence of the two functions (the ground truth and the proposed one) on a large number of randomly generated card plays.

Note that the "reasoning" and "confidence_level" fields that the player LLM is instructed to provde have no impact on the gameplay. They are  used for analysis of the model's internal state and meta-reasoning abilities. The whole json response is fed back in the prompt for the next turn, allowing the model to reflect on its previous reasoning and adjust its strategy accordingly.

## Results

### Preliminary proof of concept tournament

We use Inference provider on Hugging Face Hub to run the benchmark on various open-source LLMs.

1. GPT-OSS 120B
2. GPT-OSS 20B
3. Llama 3.1 70B


Then we created a tournament with the 3 models. Each model played 20 games (20 different rules), with a maximum of 40 turns per game.
Hands were randomly drawn at the start of each game.


```
"cumulative_scores": {
    "S3 \u2014 Llama-3.3-70B-Instruct": 268,
    "S1 \u2014 GPT OSS 120B": 187,
    "S2 \u2014 GPT OSS 20B": 189
  },
  "win_counts": {
    "S3 \u2014 Llama-3.3-70B-Instruct": 0,
    "S1 \u2014 GPT OSS 120B": 8,
    "S2 \u2014 GPT OSS 20B": 8
  }
```

Both GPT-OSS models significantly outperformed Llama 3.1 70B, with similar scores and win counts between them.

### Analysis of model behavior

We have access to the full game logs, including the models' reasoning and confidence levels at each turn. This allows us to analyze their inductive reasoning process in detail.

Here is an example output, the actual secret rule was "The absolute difference between this card’s rank and the previous card’s rank must be ≤ 2."

The state was (brackets indicate rejected cards):
```
9♠ [5♥] [Q♣] 10♠ [3♠] 10♦ J♥ Q♠ Q♥ J♠ [6♣] Q♦
```

The reasoning trace of GPT-OSS 20B ended with:
```
We must decide action: play a card or no-play. We need to provide reasoning, tentative_rule, confidence, guess_rule_if_accepted. We think rule is diff <=1. We should play a card that satisfies diff <=1. Our hand: K♦ (13), 5♦ (5), 2♥ (2), A♠ (1), A♠ (1), 6♥ (6), 6♠ (6), 6♠ (6), A♣ (1), 7♥ (7).\n\nLast mainline card: Q♦ (12). We need to play card rank within 1 of 12: 11,12,13. That includes J♠ (11), Q♦ (12), K♦ (13). We have K♦ (13). So we can play K♦. That would be accepted if rule is diff <=1. We could also play 7♥? diff 5 >1. So no. So best play: K♦.\n\nWe should guess rule if accepted? Our confidence: maybe 7? But we think rule diff <=1. But earlier guesser was wrong. But we think it\'s correct. But maybe we should be cautious. Our confidence maybe 7. We set guess_rule_if_accepted: true? We can guess if we think it\'s correct. But risk of penalty if wrong. We can still guess. We can set guess_rule_if_accepted true. Provide tentative_rule: "The rank of the new card must differ from the rank of the last mainline card by at most 1 (i.e., |new_rank – last_rank| ≤ 1)
```


```
<ACTION>
{
    "reasoning": "The last mainline card is Q♦ (rank 12). All accepted cards so far differ from the previous mainline card by at most 1 in rank, and all rejected cards differ by more than 1. Thus the rule that fits the current data is that a new card must have a rank within 1 of the last mainline card. I have K♦ (rank 13) in my hand, which satisfies this rule, so I will play it.",   
    "action": "K♦",    
    "tentative_rule": "The rank of the new card must differ from the rank of the last mainline card by at most 1 (i.e., |new_rank – last_rank| ≤ 1).", 
    "confidence_level": 7,   
    "guess_rule_if_accepted": true
}
</ACTION>'
```

We can see that the model is able to identify a plausible hypothesis about the rule, and select a card to test it. However, its tentative rule is incorrect (the actual rule allowed differences up to 2), and its confidence level is somewhat overestimated given the limited evidence.


### Larger scale benchmark and future work

More models, more rules, controlled hands, vary the rule difficulty, open vs closed models, etc.


## Bibliography

### Symbolic AI and Eleusis
Dietterich, T. G. (1980, August). Applying General Induction Methods to the Card Game Eleusis. In AAAI (pp. 218-220).

Dietterich, T. G. (1980). Multiple-Model Induction in Eleusis.

Berry, Michael JA. "APL and the search for truth: A set of functions to play New Eleusis." Proceedings of the international conference on APL. 1981.

Dietterich, Thomas G., and Ryszard S. Michalski. "Discovering patterns in sequences of events." Artificial Intelligence 25.2 (1985): 187-232.

Kurzen, L. (2010). Eleusis: Complexity and interaction in inductive inference.

Garanina, N., & Gorlatch, S. (2024). Knowledge Acquisition in Multi-Agent Systems: A Formalization of the Eleusis Card Game. Journal of Mathematical Sciences, 280(3), 468-479.

### Science education and Eleusis

Romesburg, H. C. (1979). Simulating Scientific Inquiry with the Card Game Eleusis. Science Education, 63(5), 599-608.

Dias, G., & P dos Santos, R. (2015). The game of Eleusis: an entertaining simulation of the research heuristic. Acta Scientiae (Ulbra), 17(3).

Walkup, J. R., & Key, R. (2020). Speed Eleusis: variation on an old educational card game. The Physics Teacher, 58(3), 160-163.


