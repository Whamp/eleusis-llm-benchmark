# Benchmarking in-context inductive reasoning of LLMs in the card game Eleusis

Large language models (LLMs) have shown striking in-context learning abilities, for example in few-shot settings. This has sparked renewed debate about whether they can truly infer underlying rules of the world—both in tasks resembling scientific discovery and, more generally, in forming internal “world models” that support prediction, simulation, and decision making.

The card game Eleusis is a rule-discovery game that has often been compared to the scientific method. One player, the Rule-maker, chooses a secret rule that defines which cards may be played in sequence. The other players, the Scientists, try to uncover this rule by playing cards from their hands. After each play, the Rule-maker simply states whether the card is consistent with the rule or not. At any time, a Scientist who feels confident enough can propose an explicit formulation of the secret rule to end the game.

Playing Eleusis well requires exactly the kinds of abilities we would like to probe in LLMs: forming hypotheses from small amounts of data, choosing informative “experiments” (which card to play next), revising hypotheses in light of new evidence, and assessing one’s own uncertainty (meta-reasoning). Compared to traditional LLM benchmarks, Eleusis is inherently multi-turn, open-ended, and comes with a precise ground truth. Its concept space is symbolic but natural—card ranks, colors, suits, and simple relations between them—making it a compact, controlled setting to study in-context inductive reasoning in large language models.

## Eleusis

This is a short description of the rules we have been using, which are a slightly simplified version of Eleusis. This variant uses two standard 52-card decks shuffled together.

### Setup and rule creation

One player is the Rule-maker (or game master). They secretly choose a rule that determines which cards may be played in sequence. The rule may only refer to the visible sequence and to simple properties of the cards (rank, suit, color, parity, etc.).

Example rules:
•	“Suits must alternate between spades and diamonds.”
•	“Red cards must be ranks A–10; black cards must be face cards (J, Q, K).”
•	“All ranks must be prime numbers.”

The Rule-maker then draws a first card that satisfies the rule and places it on the table to start the main line of the sequence.
Each other player starts with a hand of 12 cards. 

### Turns
Players take turns trying to play exactly one card from their hand to continue the main line, or declare a "no-play" (see below).
#### Playing a card
The Rule-maker announces whether the card obeys the rule.
If it does, the card is placed on the main line (hence the player’s hand size decreases by 1.)
If it does not, the card is placed below the last card on the main line (forming a “rejected” branch; see image below) and the player must draw 2 new cards from the deck (net hand change: +1).

#### No-play
If a player believes that none of their cards can legally be played, they may declare “no-play”.
They reveal their hand to the Rule-maker, who checks each card.
If at least one card could have been played, the Rule-maker chooses one such card, places it on the main line, and the player must draw 3 cards (net penalty +2).
If no card in the hand is legal, the player discards their entire hand and receives a new hand with 3 fewer cards than before.

#### Rule guessing
After any successful play (including a validated “no-play”), the active player may propose a rule out loud.
The Rule-maker decides whether the proposed rule is equivalent to the secret rule, in the sense that from this point on it would classify every possible future play in exactly the same way.
If the proposal is accepted as equivalent, the game ends immediately.
If no one finds an equivalent rule, the game ends after 40 turns.
### Scorekeeping

At the end of the game, each player scores the number of cards remaining in their hand (fewer is better). 

A player who correctly states an equivalent rule receives a bonus of −6 points to their score.

(In vanilla Eleusis, the Rule-maker also scores points based on how well they challenged the Scientists, but since it tests the ability to create good rules rather than to discover them, we do not include this aspect in our benchmark.)

## Benchmark
### Methodology
#### Rule creation and tournament setup
We first asked an LLM (GPT-OSS 120B) to create a set of rules, trying to create rules that are not too complicated nor too easy (in the real game, there is a scoring mechanism for the rule maker that incentives them to create rules that are a sweet spot). Then we created a tournament with 3 models : GPT-OSS 120B, GPT-OSS 20B and Llama 3.1 70B. Each model played 5 games per rule, with a maximum of 40 turns per game.

Python assessment for rule matching and guessed rule equivalence.

#### Gameplay
Inference provider
Prompted with rules, game state and hand, models were asked to choose a card to play or declare no-play.
They were asked to generate a tentative rule, how confident they are (scale 0-10) and to decide whether they wanted to try it if possible (not necessary this is a risky move since a wrong guess can be costly).




### Results
