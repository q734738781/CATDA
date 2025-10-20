"""
Module for interacting with Google Generative AI models using Langchain.
"""
import logging
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks.usage import get_usage_metadata_callback
import os

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

logger = logging.getLogger(__name__)

def get_google_model(model_name: str, **kwargs: Any) -> ChatGoogleGenerativeAI:
    """
    Initializes and returns a Google Generative AI chat model instance.

    Args:
        model_name (str): The specific Google model name (e.g., 'gemini-pro').
                          The 'google_' prefix is handled by the caller.
        **kwargs: Additional keyword arguments to pass to the ChatGoogleGenerativeAI constructor.
                  Common arguments include 'temperature', 'top_p', 'top_k'.
                  An 'api_key' can also be passed via kwargs if not set as an environment variable (GOOGLE_API_KEY).

    Returns:
        ChatGoogleGenerativeAI: An instance of the Google Generative AI chat model.

    Raises:
        ImportError: If the langchain_google_genai package is not installed.
        Exception: Any exception raised during the instantiation of ChatGoogleGenerativeAI.
    """
    try:
        # Ensure 'model' argument is passed correctly
        if 'model' not in kwargs:
             kwargs['model'] = model_name
        elif kwargs['model'] != model_name:
             logger.warning(f"Overriding 'model' kwarg ('{kwargs['model']}') with provided model_name ('{model_name}').")
             kwargs['model'] = model_name

        logger.info(f"Initializing Google model: {model_name} with kwargs: {kwargs}")
        llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, **kwargs)
        return llm
    except ImportError as e:
        logger.error("Failed to import ChatGoogleGenerativeAI. Ensure 'langchain-google-genai' is installed.")
        raise ImportError("The 'langchain-google-genai' package is required to use Google models. Please install it.") from e
    except Exception as e:
        logger.error(f"Error initializing Google model '{model_name}': {e}")
        # Re-raise the original exception for the caller to handle
        raise 

if __name__ == "__main__":
    model = get_google_model("gemini-2.5-flash-preview-04-17")
    with get_usage_metadata_callback() as usage_callback:
        response = model.invoke("What is the capital of France?")
        print(response)
        print(usage_callback.usage_metadata)

