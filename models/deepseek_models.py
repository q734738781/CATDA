"""
Deepseek model configuration module.
This module provides functions to initialize and configure Deepseek chat models.
"""

import os
from langchain_openai import ChatOpenAI
from CATDA.models.utils import check_model
DeepSeek_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
temp = 0

def get_deepseek_model(model_name: str, temperature=temp, api_key=DeepSeek_API_KEY, **kwargs):
    """Get a configured Deepseek chat model instance based on the provided model name.
    
    Args:
        model_name (str): The specific Deepseek model to use (e.g., 'deepseek-chat', 'deepseek-coder').
        temperature (float): Sampling temperature for the model
        api_key (str): API key for authentication
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        ChatOpenAI: Configured Deepseek chat model instance
    """
    
    model = ChatOpenAI(
        model=model_name,
        openai_api_base="https://api.deepseek.com/v1",
        openai_api_key=api_key,
        temperature=temperature,
        verbose=True,
        **kwargs
    )
    return model

if __name__ == "__main__":
    try:
        llm_model = get_deepseek_model(model_name="deepseek-chat")
        check_model(llm_model)
    except Exception as e:
        print(f"Error during testing: {e}")