import logging
import json # For handling Neo4j results
from typing import Type, Any, List, Dict
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import os

# Try to import neo4j and handle potential ImportError
try:
    import neo4j
    from neo4j import GraphDatabase, basic_auth
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None # Define as None if neo4j is not installed
    ServiceUnavailable = None
    AuthError = None
    NEO4J_AVAILABLE = False
    logging.warning("neo4j package not found. GraphQueryTool will not be able to connect. Run `pip install neo4j`")

logger = logging.getLogger(__name__)

# --- Updated Input Schema ---
class GraphCypherQueryInput(BaseModel):
    """Input schema for the GraphQueryTool, expecting a Cypher query."""
    cypher_query: str = Field(description="A valid Cypher query string to execute against the Neo4j database.")

class GraphQueryTool(BaseTool):
    """
    Tool to execute a provided Cypher query against the Neo4j graph containing data.
    The input to this tool MUST be a valid Cypher query string.
    """
    name: str = "GraphQuery"  # Align with spec
    description: str = (
        "Execute a read-only Cypher query against the Neo4j graph. "
        "Use for fetching specific nodes, relationships, or properties. "
        "Input MUST be a valid Cypher query string. Automatically adds LIMIT 100. Returns results as JSON."
    )
    args_schema: Type[BaseModel] = GraphCypherQueryInput # Use the new input schema

    _neo4j_uri: str = PrivateAttr()
    _neo4j_user: str = PrivateAttr()
    _neo4j_password: str = PrivateAttr()

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, **kwargs):
        """Initialize the tool with Neo4j connection details."""
        super().__init__(**kwargs)
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver is not available. Please install it using `pip install neo4j`.")
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password

    def _run(self, cypher_query: str) -> str:
        """Execute the provided Cypher query against the Neo4j database."""
        logger.info(f"GraphQueryTool received Cypher query: {cypher_query}")

        if not NEO4J_AVAILABLE:
            return "Error: Neo4j driver not installed. Cannot query graph."
        if not cypher_query:
            return "Error: Received empty Cypher query."

        # Guardrail: disallow mutation keywords
        forbidden_keywords = ["create", "merge", "set", "delete"]
        lowered = cypher_query.lower()
        if any(k in lowered for k in forbidden_keywords):
            logger.warning("Rejected query containing mutation clause.")
            return "Error: Mutation clauses (CREATE / MERGE / SET / DELETE) are not allowed."

        # Auto-append LIMIT unless query already contains a LIMIT clause (case-insensitive)
        if "limit" not in lowered:
            top_k = int(os.getenv("GRAPH_QUERY_TOP_K", "100"))
            cypher_query = f"{cypher_query.strip()} LIMIT {top_k}"
            logger.debug(f"Auto-appended LIMIT {top_k} to Cypher query.")

        driver = None # Ensure driver is defined for the finally block
        try:
            # Create the driver instance directly here
            driver = GraphDatabase.driver(self._neo4j_uri, auth=basic_auth(self._neo4j_user, self._neo4j_password))
            driver.verify_connectivity() # Verify connection right away
            logger.info(f"Neo4j driver connected successfully to {self._neo4j_uri} for query.")

            logger.info(f"Executing Cypher: {cypher_query}")
            with driver.session() as session:
                result = session.run(cypher_query)
                # Fetch all records and convert them to a list of dictionaries
                records = [record.data() for record in result]
                logger.info(f"Cypher query executed successfully. Found {len(records)} records.")

                # Serialize the result list to a JSON string for the LLM
                # Handle potential serialization errors if data is complex
                try:
                    result_json = json.dumps(records, default=str) # Use default=str for non-serializable types
                except TypeError as json_err:
                     logger.error(f"Error serializing query results to JSON: {json_err}")
                     return f"Error: Query executed, but results could not be serialized to JSON. Found {len(records)} records. Serialization error: {json_err}"

                return result_json

        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Neo4j connection error during query execution: {e}")
            return f"Error: Could not connect to Neo4j database ({e})."
        except Exception as e: # Catch potential Cypher syntax errors or other execution errors
            logger.error(f"An error occurred during Cypher query execution: {e}", exc_info=True)
            # Check if it's a known Neo4j client error (e.g., syntax error)
            if "org.neo4j.driver.exceptions" in str(type(e)):
                 return f"Error executing Cypher query: {e}"
            else:
                 return f"Error: An unexpected error occurred while querying the graph: {e}"
        finally:
             # Ensure the driver is closed after each execution
             if driver:
                 try:
                     driver.close()
                     logger.debug("Neo4j driver closed.")
                 except Exception as close_e:
                      logger.error(f"Error closing Neo4j driver: {close_e}")


    async def _arun(self, cypher_query: str) -> str:
        """Asynchronous execution (placeholder - needs async driver/query)."""
        logger.warning("_arun (async graph query) is not fully implemented yet. Falling back to sync.")
        # TODO: Implement proper async execution using neo4j.aio
        return self._run(cypher_query)

# Example usage (for testing the tool directly) - Needs update to reflect Cypher input
if __name__ == '__main__':
    import os
    logging.basicConfig(level=logging.INFO)

    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")

    if not password:
        logger.info("Error: NEO4J_PASSWORD environment variable not set. Cannot run test.")
    elif not NEO4J_AVAILABLE:
        logger.info("Error: neo4j package not installed. Cannot run test.")
    else:
        logger.info(f"Attempting to connect to Neo4j at {uri} as user {user}...")
        try:
            tool = GraphQueryTool(neo4j_uri=uri, neo4j_user=user, neo4j_password=password)

            # Test with a valid Cypher query
            test_cypher = "MATCH (p:Paper) RETURN p.name AS paper_name LIMIT 5"
            logger.info(f"\nTesting tool with Cypher: '{test_cypher}'")
            # Input must now match GraphCypherQueryInput schema
            result = tool.run({"cypher_query": test_cypher})
            logger.info(f"\nTool Result:\n{result}")

            # Test with an invalid Cypher query
            invalid_cypher = "MATCH (n RETURN n LIMIT 1"
            logger.info(f"\nTesting tool with invalid Cypher: '{invalid_cypher}'")
            result_invalid = tool.run({"cypher_query": invalid_cypher})
            logger.info(f"\nTool Result (Invalid Cypher):\n{result_invalid}")


        except (ImportError, ServiceUnavailable, AuthError) as e:
            logger.info(f"\nTest failed: Could not connect or initialize tool. Error: {e}")
        except Exception as e:
            logger.info(f"\nTest failed with unexpected error: {e}")
            import traceback
            traceback.logger.info_exc() 