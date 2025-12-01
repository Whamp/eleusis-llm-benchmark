# Prompt Design for Thinking Models

All prompts in this project follow a consistent pattern designed for thinking/reasoning models:

## Design Pattern

1. **Allow Reasoning**: Models can think through the problem before answering
2. **Clear Formatting**: Final answer wrapped in XML tags for reliable extraction
3. **Graceful Fallback**: Parser tries XML tags → code blocks → raw response

## Prompt Types

### 1. Rule Generation (`<RULE>...</RULE>`)

```
Think through your rule if needed, but your FINAL OUTPUT must end with:
<RULE>complete rule description</RULE>
```

**Extraction**: Extracts text between `<RULE>` tags, ignores reasoning before it.

---

### 2. Move Selection (`<ACTION>...</ACTION>`)

```
Think through your analysis if needed, then wrap your final decision in XML tags:

<ACTION>
{
    "action": "play_card" or "no_play",
    "card": "5♥",
    "reasoning": "..."
}
</ACTION>
```

**Extraction**: Extracts JSON from `<ACTION>` tags via `xml_tag="ACTION"` parameter.

---

### 3. Rule Guessing (`<GUESS>...</GUESS>`)

```
Think through the patterns you've observed, then wrap your final decision in XML tags:

<GUESS>
{
    "should_guess": true/false,
    "guess": "Your rule description",
    "confidence": 0-100
}
</GUESS>
```

**Extraction**: Extracts JSON from `<GUESS>` tags via `xml_tag="GUESS"` parameter.

---

### 4. Rule Evaluation (`<EVALUATION>...</EVALUATION>`)

```
Think through how your rule applies to this card, then wrap your final answer in XML tags:

<EVALUATION>
{
    "result": "in" or "out",
    "reasoning": "..."
}
</EVALUATION>
```

**Extraction**: Extracts JSON from `<EVALUATION>` tags via `xml_tag="EVALUATION"` parameter.

---

## Implementation

### Prompts (`prompts.py`)
All prompts explicitly instruct models to:
1. Think/reason as needed
2. Wrap final answer in specific XML tags
3. Include examples showing the format

### Parsing (`llm_client.py`)
The `generate_structured()` method:
1. Accepts `xml_tag` parameter
2. Searches for `<TAG>...</TAG>` pattern (case-insensitive)
3. Falls back to code blocks then raw response
4. Parses extracted JSON

### Usage
```python
# In code
response = client.generate_structured(
    prompt,
    max_tokens=2048,
    xml_tag="ACTION"  # Looks for <ACTION>...</ACTION>
)
```

## Benefits

✅ **Thinking models**: Can reason without polluting structured output
✅ **Reliability**: XML tags more robust than hoping for clean JSON
✅ **Debugging**: Full reasoning visible in `llm_interactions_*.txt` log
✅ **Consistency**: Same pattern across all prompts
