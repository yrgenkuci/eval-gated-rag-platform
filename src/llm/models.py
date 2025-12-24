"""LLM data models."""

from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A message in a conversation.

    Attributes:
        role: The role of the message sender.
        content: The message content.
    """

    role: Role = Field(description="Message role")
    content: str = Field(description="Message content")


class GenerationResult(BaseModel):
    """Result from LLM generation.

    Attributes:
        content: The generated text.
        model: Model used for generation.
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used.
    """

    content: str = Field(description="Generated text")
    model: str = Field(description="Model used")
    prompt_tokens: int = Field(default=0, description="Prompt token count")
    completion_tokens: int = Field(default=0, description="Completion token count")
    total_tokens: int = Field(default=0, description="Total token count")

