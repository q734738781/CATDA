import logging
import json
import re # For basic validation
from typing import Type, Dict, Any, List # Added List
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

# Neo4j Imports
try:
    from neo4j import GraphDatabase, basic_auth
    from neo4j.exceptions import ServiceUnavailable, AuthError, CypherSyntaxError
    NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None
    ServiceUnavailable = None
    AuthError = None
    CypherSyntaxError = None
    NEO4J_AVAILABLE = False
    logging.warning("neo4j package not found. GraphAggregateTool will not be able to connect. Run `pip install neo4j`")

logger = logging.getLogger(__name__)

# --- Tool Input Schema ---
class GraphAggregateInput(BaseModel):
    """Input schema for the GraphAggregateTool."""
    cypher_aggregation_query: str = Field(description=(
        "A valid Cypher query designed for server-side aggregation (using functions like avg(), min(), max(), stdev(), count()). "
        "The query MUST start with 'MATCH' or 'OPTIONAL MATCH' and MUST NOT contain mutation clauses (CREATE, MERGE, SET, DELETE, REMOVE)."
    ))

class GraphAggregateTool(BaseTool):
    """
    Executes a provided Cypher query designed for server-side aggregation
    (e.g., calculating average, min, max) against the Neo4j graph.
    Includes basic validation to prevent graph mutations.
    Returns the single row of aggregated results as a dictionary.
    """
    name: str = "GraphAggregate"
    description: str = (
        "Execute a read-only Cypher query containing aggregation functions (e.g., avg, min, max, count, stdev) "
        "directly on the server for efficiency. Input MUST be a valid Cypher query with aggregations. Returns a single result dictionary."
    )
    args_schema: Type[BaseModel] = GraphAggregateInput

    # Neo4j connection
    _neo4j_uri: str = PrivateAttr()
    _neo4j_user: str = PrivateAttr()
    _neo4j_password: str = PrivateAttr()

    # Validation settings
    _allowed_start_keywords: List[str] = PrivateAttr(default=["MATCH", "OPTIONAL MATCH"])
    _forbidden_keywords: List[str] = PrivateAttr(default=["CREATE", "MERGE", "SET", "DELETE", "REMOVE"])
    _required_aggregation_functions: List[str] = PrivateAttr(default=["avg(", "min(", "max(", "stdev(", "count(", "sum(", "percentileCont(", "percentileDisc("]) # Common aggregation functions

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, **kwargs):
        """Initialize with Neo4j connection details."""
        super().__init__(**kwargs)
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver is not available.")
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password

    def _validate_query(self, query: str) -> bool:
        """Performs basic validation on the Cypher query."""
        query_upper = query.strip().upper()

        # Check if starts with allowed keywords
        if not any(query_upper.startswith(keyword) for keyword in self._allowed_start_keywords):
            logger.error(f"Validation failed: Query does not start with allowed keywords ({self._allowed_start_keywords}). Query: {query}")
            return False

        # Check for forbidden keywords (simple substring check, might need improvement for robustness)
        if any(keyword in query_upper for keyword in self._forbidden_keywords):
             logger.error(f"Validation failed: Query contains forbidden mutation keyword ({self._forbidden_keywords}). Query: {query}")
             return False

        # Check if it contains at least one aggregation function (case-insensitive)
        # This helps ensure it's likely an aggregation query as intended
        query_lower = query.lower()
        if not any(func in query_lower for func in self._required_aggregation_functions):
             logger.warning(f"Validation warning: Query does not appear to contain common aggregation functions ({self._required_aggregation_functions}). Query: {query}")
             # Allow execution but log a warning, as it might be a valid but unusual aggregation

        return True

    def _run(self, cypher_aggregation_query: str) -> Dict[str, Any]:
        """Executes the validated Cypher aggregation query."""
        logger.info(f"Received aggregation query: {cypher_aggregation_query}")
        if not NEO4J_AVAILABLE:
            return {"error": "Neo4j driver not installed."}
        if not cypher_aggregation_query:
            return {"error": "Received empty Cypher query."}

        # Validate the query before execution
        if not self._validate_query(cypher_aggregation_query):
            return {"error": "Invalid aggregation query: Does not meet validation criteria (must start with MATCH, contain aggregation, no mutations)."}

        driver = None
        try:
            driver = GraphDatabase.driver(self._neo4j_uri, auth=basic_auth(self._neo4j_user, self._neo4j_password))
            driver.verify_connectivity()
            logger.info(f"Executing aggregation Cypher: {cypher_aggregation_query}")

            with driver.session() as session:
                result = session.run(cypher_aggregation_query)
                # Aggregation queries typically return a single record
                record = result.single()

                if record:
                    # Convert the single record to a dictionary
                    result_dict = record.data()
                    logger.info(f"Aggregation successful. Result: {result_dict}")
                    # Serialize potentially complex types (like Neo4j Dates/Times) just in case
                    try:
                         # Use json.dumps with default=str and then json.loads to ensure basic types
                         return json.loads(json.dumps(result_dict, default=str))
                    except TypeError as json_err:
                          logger.error(f"Error serializing aggregation results: {json_err}")
                          return {"error": f"Query executed, but result serialization failed: {json_err}", "raw_result": str(result_dict)}

                else:
                    logger.warning("Aggregation query returned no results.")
                    # Return empty dict instead of warning message for consistency
                    return {}


        except CypherSyntaxError as e:
             logger.error(f"Cypher syntax error during aggregation: {e}")
             return {"error": f"Cypher syntax error: {e}"}
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Neo4j connection error during aggregation: {e}")
            return {"error": f"Neo4j connection error: {e}"}
        except Exception as e:
            logger.error(f"An unexpected error occurred during aggregation: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}
        finally:
            if driver:
                driver.close()

    async def _arun(self, cypher_aggregation_query: str) -> Dict[str, Any]:
        """Asynchronous execution (placeholder)."""
        logger.warning("_arun (async graph aggregation) is not implemented. Falling back to sync.")
        return self._run(cypher_aggregation_query)

# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import os

    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")

    if not password:
        logger.info("Error: NEO4J_PASSWORD environment variable not set.")
    elif not NEO4J_AVAILABLE:
        logger.info("Error: neo4j package not installed.")
    else:
        logger.info(f"Testing GraphAggregateTool connection to {uri}...")
        try:
            tool = GraphAggregateTool(neo4j_uri=uri, neo4j_user=user, neo4j_password=password)

            # Test 1: Valid aggregation query (replace with relevant query for your graph)
            valid_agg_query = "MATCH (t:Testing) RETURN avg(t.temperature_C) AS avg_temp, count(t) AS num_tests"
            logger.info(f"\nTesting valid query: {valid_agg_query}")
            result1 = tool.run({"cypher_aggregation_query": valid_agg_query})
            logger.info(f"Result 1: {result1}")

            # Test 2: Invalid query (contains MERGE)
            invalid_query_mutate = "MATCH (p:Paper {name:'Test'}) MERGE (a:Author {name:'New'}) RETURN count(a)"
            logger.info(f"\nTesting invalid query (mutation): {invalid_query_mutate}")
            result2 = tool.run({"cypher_aggregation_query": invalid_query_mutate})
            logger.info(f"Result 2: {result2}") # Expected: error

            # Test 3: Invalid query (doesn't start with MATCH)
            invalid_query_start = "RETURN count(*)"
            logger.info(f"\nTesting invalid query (start): {invalid_query_start}")
            result3 = tool.run({"cypher_aggregation_query": invalid_query_start})
            logger.info(f"Result 3: {result3}") # Expected: error

            # Test 4: Valid query without common aggregation (warning expected)
            valid_but_unusual = "MATCH (c:Catalyst) RETURN collect(c.name)[0..5] AS sample_names"
            logger.info(f"\nTesting valid but unusual query: {valid_but_unusual}")
            result4 = tool.run({"cypher_aggregation_query": valid_but_unusual})
            logger.info(f"Result 4: {result4}")


        except (ImportError, ConnectionError, RuntimeError) as e:
            logger.info(f"\nTest failed: {e}")
        except Exception as e:
            logger.info(f"\nTest failed with unexpected error: {e}")
            import traceback
            traceback.logger.info_exc() 