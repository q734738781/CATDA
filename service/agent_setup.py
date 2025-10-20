import logging
import sys
import os

# Langchain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Project imports - Adjusted path assuming this file is in the service directory
# Using relative imports now
try:
    # Assumes models and agentic_tools are in sibling directories or accessible via project root in sys.path
    from ..models.models import get_model
    # Core tools (group A)
    from ..agentic_tools.graph_query_tool import GraphQueryTool
    from ..agentic_tools.graph_schema_tool import GraphSchemaTool
    from ..agentic_tools.name_resolver_tool import NameResolverTool
    from ..agentic_tools.fieldname_resolver_tool import FieldNameResolverTool
    from ..agentic_tools.evidence_fetcher_tool import EvidenceFetcherTool
    from ..agentic_tools.web_search_tool import WebSearchTool
    # Additional tools (group B)
    from ..agentic_tools.unit_converter_tool import UnitConverterTool
    from ..agentic_tools.synthesis_path_retriever import SynthesisPathRetrieverTool
    # Prompt
    from ..prompts.agent_prompt import agent_system_prompt as system_prompt
except ImportError as e:
    # If run directly or package structure is wrong, this will fail.
    logger = logging.getLogger(__name__) # Need logger for error message
    logger.error(f"Error importing project modules in agent_setup using relative paths: {e}")
    raise


logger = logging.getLogger(__name__)


def setup_agent(
    model_name: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    verbose: bool = False,
    name_regex_map_path: str | None = None,
    field_regex_map_path: str | None = None,
) -> AgentExecutor:
    """
    Initializes the LLM, tools, prompt, agent, and executor.
    Args:
        model_name: Name of the language model to use.
        neo4j_uri: Connection URI for the Neo4j database.
        neo4j_user: Username for Neo4j database.
        neo4j_password: Password for Neo4j database.
        verbose: Whether to run the agent executor in verbose mode.

    Returns:
        An initialized AgentExecutor instance.

    Raises:
        SystemExit: If model initialization or tool initialization fails.
    """
    logger.info(f"Initializing LLM: {model_name}")
    # Ensure API keys are set as environment variables (e.g., DEEPSEEK_API_KEY or OPENAI_API_KEY)
    # The get_model function should handle underlying client initialization
    try:
        # Add any necessary kwargs for the specific model if needed
        # Example: llm = get_model(model_name, temperature=0.1)
        llm = get_model(model_name)
    except Exception as e:
        logger.error(f"Failed to initialize model '{model_name}': {e}")
        logger.error("Ensure the required API key is set as an environment variable (e.g., DEEPSEEK_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY).", exc_info=True)
        sys.exit(1)

    logger.info("Initializing tools...")
    cache_dir = os.environ.get("CAT_EXTRACTOR_CACHE_DIR", os.path.join(os.getcwd(), ".cache"))
    # Initialize each tool with connection details or other parameters
    tools = []
    try:
        graph_query_tool = GraphQueryTool(neo4j_uri=neo4j_uri, neo4j_user=neo4j_user, neo4j_password=neo4j_password)
        graph_schema_tool = GraphSchemaTool(neo4j_uri=neo4j_uri, neo4j_user=neo4j_user, neo4j_password=neo4j_password, cache_dir=cache_dir)
        name_resolver_tool = NameResolverTool(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            # node_labels_for_index=["Catalyst", "Chemical"], # Optionally filter index
            cache_dir=cache_dir,
            regex_map_path=name_regex_map_path,
        )
        fieldname_resolver_tool = FieldNameResolverTool(
            neo4j_uri=neo4j_uri, # Pass connection details for dynamic key fetching
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            cache_dir=cache_dir,
            regex_map_path=field_regex_map_path,
        )
        evidence_fetcher_tool = EvidenceFetcherTool(neo4j_uri=neo4j_uri, neo4j_user=neo4j_user, neo4j_password=neo4j_password)
        unit_converter_tool = UnitConverterTool() # No Neo4j needed
        web_search_tool = WebSearchTool() # Uses TAVILY_API_KEY from env
        synthesis_path_retriever_tool = SynthesisPathRetrieverTool(neo4j_uri=neo4j_uri, neo4j_user=neo4j_user, neo4j_password=neo4j_password)

        tools = [
            name_resolver_tool,
            fieldname_resolver_tool,
            evidence_fetcher_tool,
            synthesis_path_retriever_tool,
            graph_schema_tool,
            graph_query_tool,
            unit_converter_tool,
            web_search_tool,
        ]
    except ImportError as e:
        logger.error(f"Failed to initialize one or more tools due to missing dependencies: {e}")
        logger.error("Please ensure all required libraries (e.g., neo4j, pint, faiss-cpu, sentence-transformers) are installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize tools: {e}", exc_info=True)
        sys.exit(1)

    logger.info("All tools initialized:")
    for t in tools:
        logger.info(f"- {t.name}: {t.description}")


    logger.info("Creating prompt...")
    # Define the prompt template
    # Ensure you have 'agent_scratchpad' for intermediate steps if using certain agent types
    # create_tool_calling_agent uses 'agent_scratchpad' implicitly.
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                # Format the system prompt with the actual tool name and description
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
    except Exception as e:
        logger.error(f"Failed to create prompt template: {e}", exc_info=True)
        sys.exit(1)


    logger.info("Creating agent...")
    try:
        # The LLM uses the tool descriptions in the prompt to decide which tool to call.
        agent = create_tool_calling_agent(llm, tools, prompt)
    except Exception as e:
        logger.error(f"Failed to create tool calling agent: {e}", exc_info=True)
        sys.exit(1)


    logger.info("Creating agent executor...")
    try:
        # Pass verbose flag from function arguments
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
    except Exception as e:
        logger.error(f"Failed to create agent executor: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Agent setup complete.")
    return agent_executor 