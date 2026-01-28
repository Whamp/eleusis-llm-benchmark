# Scoring System Analysis

## Current Scoring Formula

From `src/eleusis/game/engine.py:147-162`:

```python
def calculate_score(self, max_turns: int, current_turn: int) -> int:
    if self.rule_guessed:
        score = max_turns - current_turn - (self.wrong_guess_penalty * self.failed_guess_count)
    else:
        score = 0  # No penalty for unsuccessful rounds
    return score
```

## The Problem

**Unsuccessful rounds always score 0**, regardless of how many failed guesses occurred. This creates a perverse incentive structure:

| Scenario | Failed Guesses | Score |
|----------|----------------|-------|
| Unsuccessful, gave up early | 0 | 0 |
| Unsuccessful, tried hard | 22 | 0 |
| Successful at turn 25 | 14 | -23 |
| Successful at turn 29 | 2 | -3 |

### Empirical Evidence

From the 260121_78_rounds dataset:
- **143 unsuccessful rounds**: All have score = 0, despite 0-22 failed guesses each
- **67 rounds with negative scores**: All are successful rounds with high penalties
- Worst case: Claude Haiku round 11 scored **-23** (successful at turn 25 with 14 failed guesses)

### Incentive Analysis

The current system rewards:
1. **Never guessing** unless 100% confident (unsuccessful = 0, safe)
2. **Giving up early** rather than trying with uncertainty
3. **Extreme caution** over exploration

It punishes:
1. **Aggressive exploration** (many guesses to test hypotheses)
2. **Winning late with mistakes** (can score worse than not winning at all)

## Impact on Analysis

### No-Stakes Analysis

The "no_stakes" metric attempts to show scores if:
1. Guessing was systematic (score at first correct answer)
2. Wrong guesses had zero penalty

But for unsuccessful rounds, there's **already no penalty** - they score 0 regardless. So:

- **Successful rounds**: `no_stakes = score + 2*failed + early_correct` (removing penalty, scoring earlier)
- **Unsuccessful rounds**: `no_stakes = max_turns - first_correct + 1` if correct shadow existed, else 0

The improvement from no_stakes only comes from successful rounds, because unsuccessful rounds already have "no stakes" built in.

### Per-Model Impact

Models with aggressive guessing strategies (high failed_guesses) are penalized only when they succeed:

| Model | Avg Failed | Success Rate | Avg Score |
|-------|------------|--------------|-----------|
| Claude Haiku 4.5 | 7.55 | 70.5% | 9.14 |
| Gpt Oss 20B | 6.21 | 71.8% | 9.91 |
| Gpt 5.2 High | 0.33 | 96.2% | 14.13 |

Claude Haiku's aggressive strategy hurts it when it wins (heavy penalties), but those same failed guesses cost nothing when it loses.

## Suggested Fix

Consider penalizing unsuccessful rounds proportionally:

```python
def calculate_score(self, max_turns: int, current_turn: int) -> int:
    if self.rule_guessed:
        score = max_turns - current_turn - (self.wrong_guess_penalty * self.failed_guess_count)
    else:
        # Option A: Same penalty structure
        score = 0 - (self.wrong_guess_penalty * self.failed_guess_count)

        # Option B: Reduced penalty (failed guesses still cost something)
        # score = 0 - (self.wrong_guess_penalty * self.failed_guess_count) // 2
    return score
```

This would:
1. Make failed guesses always costly (consistent incentives)
2. Reward conservative play when uncertain
3. Make no_stakes analysis more meaningful (actual penalty removal)
4. Better differentiate models by decision-making quality

## Verification

Run this to see the current scoring behavior:

```python
import json
from pathlib import Path

for subfolder in Path('results/260121_78_rounds').iterdir():
    if subfolder.is_dir() and subfolder.name.startswith('solo_evaluation_'):
        with open(subfolder / 'results.json') as f:
            result = json.load(f)
        for rd in result['rounds']:
            if not rd['success'] and rd['failed_guesses'] > 10:
                print(f"{result['config']['player']}: unsuccessful, {rd['failed_guesses']} failed, score={rd['score']}")
```
