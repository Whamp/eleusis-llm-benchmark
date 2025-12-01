# How to Run the Eleusis Game

## Setup

1. **Set your HuggingFace token:**
```bash
export HF_TOKEN="your_token_here"
```

2. **Configure models** in `config.yaml`:
```yaml
models:
  rule_maker:
    name: "openai/gpt-oss-120b"
    temperature: 0.8
    display_name: "GPT OSS 120B"
  scientist_1:
    name: "openai/gpt-oss-20b"
    temperature: 0.7
    display_name: "GPT OSS 20B"
  # ... etc
```

## Running the Game

```bash
uv run python play_game.py
```

This creates a log file `game_log_TIMESTAMP.txt` with the complete game flow.

## Config Options

In `config.yaml`:

```yaml
game:
  max_turns: 20              # Maximum turns before game ends
  scientist_guess_threshold: 8  # Turns before considering guessing
  pause_after_turn: true     # Wait for user input after each turn (false for automated)
```

## Game Output

The game log includes:
- Turn-by-turn play with compact board notation
- Cards played and results (ACCEPTED ✓ or REJECTED ✗)
- Rule guesses and outcomes
- Final scores

### Compact Board Notation

The board is displayed using compact notation:
- `2♠ K♠ 5♥` - Accepted cards (mainline)
- `[5♠]` - Rejected cards shown in brackets
- Example: `2♠ [5♠] K♠` means "2♠ accepted, then 5♠ rejected, then K♠ accepted"

### Turn Flow

1. Scientist decides: play card or no-play
2. Rule-maker evaluates the card
3. Result: ACCEPTED ✓ or REJECTED ✗
4. Optional: Scientist guesses the rule

## Thinking Models Support

All prompts support thinking/reasoning models using XML tag pattern:

1. **Rule Generation**: `<RULE>actual rule</RULE>`
   - Model can reason, then wrap rule in tags
   - Only text inside tags is used

2. **Move Selection**: `<ACTION>{json}</ACTION>`
   - Model analyzes game state, then wraps decision in tags

3. **Rule Guessing**: `<GUESS>{json}</GUESS>`
   - Model examines patterns, then wraps guess in tags

4. **Rule Evaluation**: `<EVALUATION>{json}</EVALUATION>`
   - Model applies its rule, then wraps verdict in tags

See `PROMPTS_DESIGN.md` for details on the prompt architecture.

## Common Issues

**LLM inconsistencies:**
- Rule-maker may not follow its own rule consistently
- Try adjusting temperature (lower = more deterministic)
- Check if rule is too complex or ambiguous

**Connection errors:**
- Verify HF_TOKEN is set correctly
- Check internet connection
- Some models may be unavailable or require specific permissions
