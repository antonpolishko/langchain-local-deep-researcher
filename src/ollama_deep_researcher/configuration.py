import os
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal

from langchain_core.runnables import RunnableConfig


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"


class Configuration(BaseModel):
    """The configurable fields for the research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform",
    )
    local_llm: str = Field(
        default="llama3.2",
        title="LLM Model Name",
        description="Name of the LLM model to use",
    )
    llm_provider: Literal["ollama", "lmstudio", "openai", "anthropic", "gemini"] = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider for the LLM (Ollama, LMStudio, OpenAI, Anthropic, or Gemini)",
    )
    search_api: Literal["perplexity", "tavily", "duckduckgo", "searxng"] = Field(
        default="duckduckgo", title="Search API", description="Web search API to use"
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/",
        title="Ollama Base URL",
        description="Base URL for Ollama API",
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )
    use_tool_calling: bool = Field(
        default=False,
        title="Use Tool Calling",
        description="Use tool calling instead of JSON mode for structured output",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        title="OpenAI API Key",
        description="API key for OpenAI (if using OpenAI provider)",
    )
    openai_model: str = Field(
        default="gpt-5.2",
        title="OpenAI Model",
        description="OpenAI model name to use",
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        title="Anthropic API Key",
        description="API key for Anthropic (if using Anthropic provider)",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        title="Anthropic Model",
        description="Anthropic model name to use",
    )
    google_api_key: Optional[str] = Field(
        default=None,
        title="Google API Key",
        description="API key for Google AI (if using Gemini provider)",
    )
    gemini_model: str = Field(
        default="gemini-3-flash-preview",
        title="Gemini Model",
        description="Gemini model name to use",
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
