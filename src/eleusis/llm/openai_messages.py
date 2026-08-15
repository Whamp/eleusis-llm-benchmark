"""Conversion from provider-neutral messages to OpenAI chat parameters."""

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from eleusis.llm.base import LLMMessage


def build_openai_chat_messages(
    messages: list[LLMMessage],
) -> list[ChatCompletionMessageParam]:
    """Convert neutral chat roles into OpenAI's discriminated message union."""
    chat_messages: list[ChatCompletionMessageParam] = []
    for message in messages:
        if message["role"] == "assistant":
            chat_message = ChatCompletionAssistantMessageParam(
                role="assistant", content=message["content"]
            )
        elif message["role"] == "system":
            chat_message = ChatCompletionSystemMessageParam(
                role="system", content=message["content"]
            )
        else:
            chat_message = ChatCompletionUserMessageParam(
                role="user", content=message["content"]
            )
        chat_messages.append(chat_message)
    return chat_messages
