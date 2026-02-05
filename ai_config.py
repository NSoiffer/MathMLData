# a_config.py
import os
from dataclasses import dataclass
from typing import Any, Literal
from openai import OpenAI, AzureOpenAI
import google.genai as genai


@dataclass
class ModelConfig:
    provider: str
    model: str                # generation model
    embedding_model: str      # embeddings model
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] = "auto"

    client: Any = None

    openai_api_key: str | None = None
    azure_api_key: str | None = None
    azure_endpoint: str | None = None
    azure_api_version: str = "2024-02-15-preview"
    gemini_api_key: str | None = None


def create_client(config: ModelConfig) -> Any:
    if config.provider == "openai":
        return OpenAI(api_key=config.openai_api_key)

    if config.provider == "azure":
        if config.azure_endpoint is None:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for Azure provider")

        return AzureOpenAI(
            api_key=config.azure_api_key,
            azure_endpoint=config.azure_endpoint,  # now guaranteed str
            api_version=config.azure_api_version,
        )

    if config.provider == "gemini":
        return genai.Client(api_key=config.gemini_api_key)

    raise ValueError(f"Unknown provider: {config.provider}")


def build_config_from_cli(provider: str, ai_service_tier: str) -> ModelConfig:
    provider = provider.lower()
    service_tier: Literal[
        "auto", "default", "flex", "scale", "priority"
    ] = ai_service_tier.lower()  # type: ignore[assignment]

    # -------------------------
    # OpenAI
    # -------------------------
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider.")

        config = ModelConfig(
            provider="openai",
            model="gpt-5.2",                     # generation model
            embedding_model="text-embedding-3-large",  # embeddings model
            openai_api_key=api_key,
            service_tier=service_tier,
        )

    # -------------------------
    # Azure OpenAI
    # -------------------------
    elif provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required for Azure provider.")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required for Azure provider.")

        config = ModelConfig(
            provider="azure",
            model="gpt-5.2",                 # your Azure *generation* deployment
            embedding_model="text-embedding-3-large-deployment",  # your Azure *embedding* deployment
            azure_api_key=api_key,
            azure_endpoint=endpoint,
            service_tier=service_tier,
            )

    # -------------------------
    # Gemini (google.genai)
    # -------------------------
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        # api_key = os.getenv("GEMINI_PAID_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Gemini provider.")

        config = ModelConfig(
            provider="gemini",
            model="gemini-2.5-pro",   # generation model
            # model="gemini-3-pro-preview",   # generation model
            embedding_model="",   # Gemini does NOT support embeddings
            gemini_api_key=api_key,
            service_tier=service_tier,  # not used for Gemini
            )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Attach the client instance
    config.client = create_client(config)
    return config
