from langchain_openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_ollama import OllamaLLM
from typing import Literal, Optional

LLMProvider = Literal["openai", "anthropic", "google"]

DEFAULT_PROVIDER: LLMProvider = "openai"
DEFAULT_MODEL = "gpt-4-turbo-preview"  # Default model for OpenAI

LLM_CONFIGS = {
    "openai": {
        "class": OpenAI,
        "default_model": "gpt-4o-mini",
        "required_params": ["model"],
        "optional_params": ["temperature", "max_tokens"],
    },
    "google": {
        "class": GoogleGenerativeAI,
        "default_model": "gemini-2.0-flash",
        "required_params": ["model"],
        "optional_params": ["temperature", "max_tokens"],
    },
    "ollama": {
        "class": OllamaLLM,
        "default_model": "deepseek-r1:8b",
        "required_params": ["model"],
        "optional_params": ["temperature", "max_tokens"],
    },
}


def get_llm(
    provider: LLMProvider = DEFAULT_PROVIDER, model: Optional[str] = None, **kwargs
) -> OpenAI | GoogleGenerativeAI | OllamaLLM:
    if provider not in LLM_CONFIGS:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    config = LLM_CONFIGS[provider]
    llm_class = config["class"]

    model = model or config["default_model"]

    return llm_class(model=model, **kwargs)
