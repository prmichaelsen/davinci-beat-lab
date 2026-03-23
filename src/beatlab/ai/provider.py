"""LLM provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send a prompt and return the response text.

        Args:
            system: System prompt.
            user: User prompt.

        Returns:
            The LLM's response text.
        """
        ...


class AnthropicProvider(LLMProvider):
    """Claude provider using the Anthropic SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for --ai mode.\n"
                "Install with: pip install davinci-beat-lab[ai]"
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
