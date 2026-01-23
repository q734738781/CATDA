"""
OpenAI/OpenRouter model configuration module.
This module provides functions to initialize and configure OpenAI-compatible chat models.
"""

import os
from langchain_openai import ChatOpenAI
from openai import OpenAI

OpenAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

def get_openai_model(model_name: str, temp=0, **kwargs):
    """Get a configured OpenAI chat model instance based on the provided model name.

    Args:
        model_name (str): The specific OpenAI model to use (e.g., 'gpt-4o', 'gpt-4-turbo').
        temp (float): Sampling temperature for the model.
        **kwargs: Additional arguments to pass to the ChatOpenAI constructor.

    Returns:
        ChatOpenAI: Configured OpenAI chat model instance.

    Raises:
        ValueError: If the OpenAI API key is not found.
        # The ChatOpenAI constructor might raise other errors (e.g., authentication, invalid model).
    """
    if not OpenAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    model = ChatOpenAI(
        model=model_name, 
        temperature=temp,
        api_key=OpenAI_API_KEY,
        **kwargs,

    )
    return model


def get_openrouter_model(model_name: str, temp=0, **kwargs):
    """Get a configured OpenRouter chat model instance (OpenAI-compatible).

    Args:
        model_name (str): The specific OpenRouter model (e.g., 'openai/gpt-4o-mini').
        temp (float): Sampling temperature for the model.
        **kwargs: Additional arguments to pass to the ChatOpenAI constructor.

    Returns:
        ChatOpenAI: Configured OpenRouter chat model instance.

    Raises:
        ValueError: If the OPENROUTER_API_KEY is not found.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    base_url = kwargs.pop("base_url", OPENROUTER_BASE_URL)
    model = ChatOpenAI(
        model=model_name,
        temperature=temp,
        api_key=OPENROUTER_API_KEY,
        base_url=base_url,
        **kwargs,
    )
    return model

def get_available_models():
    """Get a list of available models from the OpenAI API."""
    if not OpenAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set. Cannot fetch available models.")
        return []
    try:
        client = OpenAI(api_key=OpenAI_API_KEY)
        models = client.models.list()
        # Filter for GPT models if desired, or return all
        # Example: return [m.id for m in models.data if 'gpt' in m.id]
        return [m.id for m in models.data]
    except Exception as e:
        print(f"Error fetching models from OpenAI: {e}")
        return []

if __name__ == "__main__":
    try:
        test_model_name = "gpt-5-mini"
        llm_model = get_openai_model(model_name=test_model_name, temp=0.1)
        response = llm_model.invoke("What is the capital of France?")
        print(f"Model response: {getattr(response, 'content', response)}")
    except Exception as e:
        print(f"Error during testing: {e}")
