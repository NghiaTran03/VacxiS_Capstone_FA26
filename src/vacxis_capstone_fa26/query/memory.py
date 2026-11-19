"""Stateless selection of bounded, contiguous conversation context."""

from collections.abc import Sequence

from .schemas import ConversationMemoryConfig, ConversationMessage


def limit_history(
    history: Sequence[ConversationMessage], config: ConversationMemoryConfig,
) -> tuple[ConversationMessage, ...]:
    """Select whole recent messages without jumping over a budget boundary."""
    messages = tuple(history)
    for message in messages:
        if not isinstance(message, ConversationMessage):
            raise TypeError("History must contain ConversationMessage objects.")
    selected = []
    characters = 0
    for message in reversed(messages):
        if len(selected) >= config.max_messages or characters + len(message.content) > config.max_characters:
            break
        selected.append(message)
        characters += len(message.content)
    selected.reverse()
    while selected and selected[0].role == "assistant":
        selected.pop(0)
    return tuple(selected)
