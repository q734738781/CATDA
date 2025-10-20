import argparse
import logging
import sys
import os
from pathlib import Path

# --- Configuration ---
DEFAULT_MODEL = 'google_gemini-2.5-pro'


# --- Logging Setup ---
# Basic config first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("launch_gradio") # Logger for this script

# Suppress overly verbose logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
# Optionally suppress Langchain/Gradio info logs unless debugging
# logging.getLogger("langchain").setLevel(logging.WARNING)
# logging.getLogger("langchain_core").setLevel(logging.WARNING)
# logging.getLogger("gradio").setLevel(logging.WARNING)

# Import Gradio app launch function AFTER path setup
# Imports should work now if run as a module (python -m CATDA.launch_gradio)
# or if the parent directory is in PYTHONPATH.
try:
    # These imports assume the parent directory is in sys.path OR running with -m
    from CATDA.ui.gradio_app import launch_ui
    # Try to import set_debug_mode if needed for debug logging
    try:
        from CATDA.models.models import set_debug_mode
        has_set_debug_mode = True
    except ImportError:
        logger.warning("set_debug_mode function not found in models. Debug logging setup might be limited.")
        has_set_debug_mode = False
except ImportError as e:
    logger.error(f"Failed to import Gradio app or models: {e}")
    logger.error("Ensure the script is run correctly (e.g., `python -m CATDA.launch_gradio` from the parent directory) and dependencies are installed.")
    logger.error(f"Current sys.path: {sys.path}")
    sys.exit(1)


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Launch Gradio UI for querying catalysis data.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (e.g., 'openai_gpt-4o', 'google_gemini-2.5-flash-preview-04-17'). Default: {DEFAULT_MODEL}"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (sets log level to DEBUG and enables agent verbosity)."
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=os.environ.get("NEO4J_URI", "neo4j://localhost:7687"),
        help="Neo4j URI (default: neo4j://localhost:7687 or NEO4J_URI env var)."
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username (default: neo4j or NEO4J_USER env var)."
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=os.environ.get("NEO4J_PASSWORD"),
        help="Neo4j password (reads NEO4J_PASSWORD env var by default). Requires env var or arg."
    )
    parser.add_argument(
        "--gradio-port",
        type=int,
        default=6810,
        help="Port to launch the Gradio app on (default: 6810)."
    )
    parser.add_argument(
        "--listen-all",
        action="store_true",
        help="Listen on 0.0.0.0 (all interfaces) instead of 127.0.0.1 (localhost)."
    )
    parser.add_argument(
        "--name-regex-map",
        type=str,
        default=os.environ.get("NAME_RESOLVER_REGEX_MAP"),
        help="Optional JSON file path for NameResolver regex mappings (env: NAME_RESOLVER_REGEX_MAP)."
    )
    parser.add_argument(
        "--field-regex-map",
        type=str,
        default=os.environ.get("FIELD_RESOLVER_REGEX_MAP"),
        help="Optional JSON file path for FieldNameResolver regex mappings (env: FIELD_RESOLVER_REGEX_MAP)."
    )

    args = parser.parse_args()

    # --- Debug Mode Setup ---
    if args.debug:
        logger.info("Debug mode enabled.")
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger level
        # Set specific loggers to DEBUG if needed (e.g., your package)
        logging.getLogger("CATDA").setLevel(logging.DEBUG)
        # Enable Langchain/Gradio debug logs if desired
        logging.getLogger("langchain").setLevel(logging.DEBUG)
        logging.getLogger("langchain_core").setLevel(logging.DEBUG)
        logging.getLogger("gradio").setLevel(logging.DEBUG)
        logger.debug("Root and CATDA loggers set to DEBUG level.")

        # Configure file logging via set_debug_mode if available
        if has_set_debug_mode:
            # Log directory relative to current working directory where script is run
            log_dir = Path("./logs")
            log_dir.mkdir(exist_ok=True)
            run_id = f"gradio_{args.model}_{os.getpid()}" # Basic run ID
            try:
                # Pass the absolute path for safety
                set_debug_mode(debug=True, output_dir=log_dir.resolve(), run_id=run_id)
                logger.info(f"Advanced debug logging potentially configured via set_debug_mode. Logs may be saved to {log_dir.resolve()} with run ID {run_id}")
            except Exception as e:
                logger.warning(f"Call to set_debug_mode failed: {e}. File logging might not work.", exc_info=True)
        else:
             logger.info("set_debug_mode not available, skipping advanced file logging setup.")

    # --- Prerequisite Checks ---
    # Check for API Keys
    required_key = None
    if args.model:
         model_provider = args.model.split('_', 1)[0].lower()
         if model_provider == 'openai':
             required_key = "OPENAI_API_KEY"
         elif model_provider == 'deepseek':
             required_key = "DEEPSEEK_API_KEY"
         elif model_provider == 'google':
             required_key = "GOOGLE_API_KEY"
         # Add checks for other providers like Anthropic etc.

         if required_key and not os.environ.get(required_key):
             logger.warning(f"Environment variable {required_key} not set for selected model provider '{model_provider}'. The application might fail if the key is required by the model setup.")
             # Consider exiting if the key is strictly necessary:
             # print(f"Error: Environment variable {required_key} is required but not set.")
             # sys.exit(1)
    else:
        logger.error("No model specified. Please use the --model argument.")
        sys.exit(1)

    # Check for Neo4j Password
    if not args.neo4j_password:
        logger.error("Neo4j password is required. Set the NEO4J_PASSWORD environment variable or provide it using --neo4j-password.")
        sys.exit(1)

    # --- Launch App ---
    logger.info(f"Launching Gradio UI with model: {args.model}")
    try:
        launch_ui(
            model_name=args.model,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            debug_mode=args.debug, # Pass debug flag to UI/Agent setup
            server_port=args.gradio_port, # Pass the port
            listen_all=args.listen_all, # Pass the listen flag
            name_regex_map_path=args.name_regex_map,
            field_regex_map_path=args.field_regex_map
        )
    except NameError as ne:
        logger.critical(f"Caught NameError during launch: {ne}. This likely means setup_agent is not correctly defined or imported.", exc_info=True)
        sys.exit(1)
    except Exception as e:
         logger.critical(f"Failed to launch Gradio UI: {e}", exc_info=True)
         sys.exit(1) 