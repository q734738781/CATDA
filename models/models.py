"""
Model factory module for getting different LLM instances.
This module provides a unified interface to access different LLM models
with optional JSON response format support.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Set higher log level for httpx and httpcore to suppress verbose output
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

# Global variables for conversation logging
DEBUG_MODE = False
OUTPUT_DIR = None
RUN_ID = None

def set_debug_mode(debug: bool = False, output_dir: Optional[Path] = None, run_id: Optional[str] = None):
    """Set debug mode and output directory for conversation logging.
    
    Args:
        debug: Whether to enable debug mode
        output_dir: Directory to save conversation logs
        run_id: Unique identifier for the current run
    """
    global DEBUG_MODE, OUTPUT_DIR, RUN_ID
    DEBUG_MODE = debug
    OUTPUT_DIR = output_dir
    RUN_ID = run_id
    logger.debug(f"Debug mode set to {debug}, output_dir: {output_dir}, run_id: {run_id}")

def get_model(model: str = 'deepseek_deepseek-chat', **kwargs: Any):
    """
    Get a model instance based on a standardized model name format using unified provider functions.

    Args:
        model (str): The model identifier in the format 'provider_officialmodelname'.
                     Examples: 'openai_gpt-4o', 'deepseek_deepseek-chat', 'openai_gpt-4-turbo'.
        **kwargs: Additional arguments to pass to the provider's model constructor function.

    Returns:
        Any: A configured language model instance (type depends on provider).

    Raises:
        ValueError: If the model name format is invalid or the provider is unknown.
        Exception: Errors raised by the underlying provider function (e.g., invalid model name).
    """
    model_lower = model.lower()
    
    try:
        provider, official_model_name = model_lower.split('_', 1)
    except ValueError:
        raise ValueError(
            f"Invalid model format: '{model}'. Expected format: 'provider_officialmodelname'."
        ) from None

    llm = None
    
    if provider == 'openai':
        # Assume .Openai has a unified function get_openai_model(model_name: str, **kwargs)
        try:
            from .openai_models import get_openai_model
            llm = get_openai_model(model_name=official_model_name, **kwargs)
        except ImportError:
             raise ImportError("Could not import 'get_openai_model' from .Openai. Ensure it exists.") from None
        except Exception as e:
             logger.error(f"Error getting OpenAI model '{official_model_name}': {e}")
             raise # Re-raise the error from the provider function

    elif provider == 'deepseek':
        # Assume .Deepseek has a unified function get_deepseek_model(model_name: str, **kwargs)
        try:
            from .deepseek_models import get_deepseek_model
            llm = get_deepseek_model(model_name=official_model_name, **kwargs)
        except ImportError:
             raise ImportError("Could not import 'get_deepseek_model' from .Deepseek. Ensure it exists.") from None
        except Exception as e:
             logger.error(f"Error getting Deepseek model '{official_model_name}': {e}")
             raise # Re-raise the error from the provider function

    # Add other providers here using the same pattern
    elif provider == 'google':
        try:
            from .google_models import get_google_model # Use the new google module
            llm = get_google_model(model_name=official_model_name, **kwargs)
        except ImportError:
             raise ImportError("Could not import 'get_google_model' from .google. Ensure it exists and langchain-google-genai is installed.") from None
        except Exception as e:
             logger.error(f"Error getting Google model '{official_model_name}': {e}")
             raise # Re-raise the error from the provider function

    # elif provider == 'anthropic':
    #     try:
    #         from .Anthropic import get_anthropic_model # Assuming this exists
    #         llm = get_anthropic_model(model_name=official_model_name, **kwargs)
    #     except ImportError:
    #          raise ImportError("Could not import 'get_anthropic_model' from .Anthropic. Ensure it exists.") from None
    #      except Exception as e:
    #          logger.error(f"Error getting Anthropic model '{official_model_name}': {e}")
    #          raise

    else:
        # Consider adding more supported providers to the error message as they are implemented
        raise ValueError(f"Unknown provider: '{provider}'. Supported providers: ['openai', 'deepseek', 'google'].")

    # This check might be redundant if provider functions always return or raise, but good practice.
    if llm is None:
         raise ValueError(f"Model instantiation failed for '{model}'. Provider function did not return a model or raise an error.")

    # Log information about the created model
    logger.info(f"Created model: {provider} / {official_model_name}")
    
    return llm

