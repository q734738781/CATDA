"""
OpenAI model configuration module.
This module provides functions to initialize and configure various OpenAI models.
"""

import os
from langchain_openai import ChatOpenAI
from CATDA.models.utils import check_model
from openai import OpenAI
OpenAI_API_KEY = os.environ.get("OPENAI_API_KEY")

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
        verbose=True,
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
    # Update the example call to reflect the new unified function
    # You need to provide a valid model name
    try:
        # Example: Use gpt-4o-mini if testing directly
        test_model_name = "gpt-4o-mini" 
        llm_model = get_openai_model(model_name=test_model_name, temp=0.1)
        print(f"Successfully created model: {test_model_name}")
        check_model(llm_model)
        
        # Example: List available models
        # print("\nAvailable OpenAI models:")
        # available = get_available_models()
        # print(available)
    except Exception as e:
        print(f"Error during testing: {e}")