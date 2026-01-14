# Model Providers: Chain-of-Thought and Token Counting

This document details how each supported model handles chain-of-thought (CoT) reasoning and token counting.

## Token Field Definitions

- **output_tokens**: Total output tokens (reasoning + answer)
- **reasoning_tokens**: Chain-of-thought/thinking tokens
- **answer_tokens**: Non-reasoning response tokens (the actual answer)

Invariant: `output_tokens = reasoning_tokens + answer_tokens`

## Summary Table

| Model | Provider | CoT Access | CoT Format | Native Output Count | Native Reasoning Count | Estimation Needed |
|-------|----------|------------|------------|---------------------|------------------------|-------------------|
| Claude Opus 4.5 | anthropic | Summarized | Separate `thinking` block | Yes (includes thinking) | No | answer_tokens |
| GPT 5.2 | openai | Hidden | N/A | Yes (includes reasoning) | Yes | None |
| Gemini 3 Pro | google | Hidden | N/A | Yes (excludes thinking) | Yes | None |
| Gemini 3 Flash | google | Hidden | N/A | Yes (excludes thinking) | Yes | None |
| DeepSeek R1 | huggingface | Full text | Inline `<think>` tags | Yes (via stream_options) | No | reasoning_tokens |
| Kimi K2 | huggingface | Full text | Separate `reasoning` field | Yes (via stream_options) | No | answer_tokens |
| GPT-OSS 120B | huggingface | Full text | Separate `reasoning` field | Yes (via stream_options) | No | answer_tokens |
| GPT-OSS 20B | huggingface | Full text | Separate `reasoning` field | Yes (via stream_options) | No | answer_tokens |
| GLM 4.7 | huggingface | Via field | `reasoning_content` field | Yes (via stream_options) | No | answer_tokens |
| Grok 4 | xai | Unknown | Unknown | Yes | Unknown | TBD |

## Detailed Provider Information

### Anthropic (Claude Opus 4.5)

**API**: Anthropic Messages API with extended thinking

**Chain-of-Thought**:
- Access: **Summarized** - the `thinking` block may contain a summary, not full reasoning
- Format: Response contains both `thinking` and `text` blocks
- See: https://platform.claude.com/docs/en/build-with-claude/extended-thinking#summarized-thinking

**Token Counting**:
- `input_tokens`: Provided by API
- `output_tokens`: Provided by API, **includes thinking tokens** (even if thinking is summarized)
- `reasoning_tokens`: Not provided - computed as output_tokens - estimated answer_tokens

**Normalization**:
```
output_tokens = API output_tokens (ground truth, includes thinking)
answer_tokens = estimate(content_text)  # word_count × 1.3
reasoning_tokens = output_tokens - answer_tokens
```

---

### OpenAI (GPT 5.2)

**API**: OpenAI Responses API with reasoning effort

**Chain-of-Thought**:
- Access: Hidden/encrypted (only summary available in some cases)
- Format: N/A - reasoning not exposed in response
- The `reasoning` field may contain a summary but not full CoT

**Token Counting**:
- `input_tokens`: Provided by API
- `output_tokens`: Provided by API, **includes reasoning tokens**
- `reasoning_tokens`: Provided by API via `output_tokens_details.reasoning_tokens`

**Normalization**:
```
output_tokens = API output_tokens
reasoning_tokens = API reasoning_tokens
answer_tokens = output_tokens - reasoning_tokens
```

---

### Google (Gemini 3 Pro/Flash)

**API**: Google GenAI with thinking config

**Chain-of-Thought**:
- Access: Hidden (thinking text not accessible)
- Format: N/A - response may have `thought` parts but content is not visible
- Thinking level can be set to low/high but content is encrypted

**Token Counting**:
- `prompt_token_count`: Provided by API
- `candidates_token_count`: Provided by API, **excludes thinking tokens**
- `thoughts_token_count`: Provided by API - actual reasoning token count

**Normalization**:
```
answer_tokens = API candidates_token_count
reasoning_tokens = API thoughts_token_count
output_tokens = answer_tokens + reasoning_tokens
```

---

### HuggingFace - DeepSeek R1

**API**: HuggingFace Inference Providers (via Together)

**Chain-of-Thought**:
- Access: Full text returned inline
- Format: Reasoning wrapped in `<think>...</think>` tags within content
- Everything before `</think>` is reasoning, everything after is the answer

**Token Counting** (streaming):
- Without `stream_options`: No token counts, must estimate from text
- With `stream_options={"include_usage": True}`: Final chunk contains usage
- `completion_tokens`: **Includes both reasoning and answer** (full content)
- `reasoning_tokens`: Not provided - must estimate from `<think>` content

**Normalization**:
```
output_tokens = API completion_tokens (or estimate if not available)
reasoning_tokens = estimate(think_content)
answer_tokens = output_tokens - reasoning_tokens
```

---

### HuggingFace - Kimi K2, GPT-OSS

**API**: HuggingFace Inference Providers (via Together)

**Chain-of-Thought**:
- Access: Full text returned in separate field
- Format: `delta.reasoning` field in streaming, separate from `delta.content`
- Reasoning and answer are cleanly separated

**Token Counting** (streaming):
- Without `stream_options`: No token counts, must estimate from text
- With `stream_options={"include_usage": True}`: Final chunk contains usage
- `completion_tokens`: **Includes both reasoning and answer** (total output)
- `reasoning_tokens`: Not provided - computed as output - estimated answer

**Normalization**:
```
output_tokens = API completion_tokens (ground truth, includes reasoning)
answer_tokens = estimate(content_text)  # word_count × 1.3
reasoning_tokens = output_tokens - answer_tokens
```

---

### HuggingFace - GLM 4.7

**API**: HuggingFace Inference Providers (via Z.AI)

**Chain-of-Thought**:
- Access: Via `reasoning_content` field (GLM-4.5 series and higher)
- Format: Response contains `reasoning_content` field separate from `content`
- Z.AI API supports `thinking` parameter to control CoT (enabled/disabled)
- See: https://docs.z.ai/api-reference/llm/chat-completion#assistant-message

**Token Counting** (streaming):
- With `stream_options={"include_usage": True}`: Final chunk contains usage
- `completion_tokens`: **Includes both reasoning and answer** (total output)
- `reasoning_tokens`: Not provided - computed as output - estimated answer

**Normalization**:
```
output_tokens = API completion_tokens (ground truth, includes reasoning)
answer_tokens = estimate(content_text)  # word_count × 1.3
reasoning_tokens = output_tokens - answer_tokens
```

**Note**: The `reasoning_content` field availability may depend on the HuggingFace Inference Providers routing. Token counts reflect reasoning effort even if content is not exposed.

---

## Estimation Method

When native token counts are not available, we estimate using:

```python
estimated_tokens = int(word_count * 1.3)
```

This assumes approximately 1.3 tokens per word on average, which is a reasonable approximation for English text with typical LLM tokenizers.

## Configuration Reference

Model configurations are defined in `models.yaml`:

```yaml
# Example: DeepSeek R1 uses think_tags format
deepseek-r1:
  provider: huggingface
  model_id: deepseek-ai/DeepSeek-R1
  hf_provider: together
  reasoning_format: think_tags  # <think>...</think> in content

# Example: Kimi K2 uses separate_field format
kimi-k2:
  provider: huggingface
  model_id: moonshotai/Kimi-K2-Thinking
  hf_provider: together
  reasoning_format: separate_field  # reasoning in delta.reasoning

# Example: GLM 4.7 uses separate_field format
glm-4.7:
  provider: huggingface
  model_id: zai-org/GLM-4.7
  hf_provider: zai-org
  reasoning_format: separate_field  # reasoning in delta.reasoning_content
```
