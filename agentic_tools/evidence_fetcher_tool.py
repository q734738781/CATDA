import logging
import json
from typing import Type, List, Dict, Any
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
    logging.warning("neo4j package not found. EvidenceFetcherTool will not be able to connect. Run `pip install neo4j`")

logger = logging.getLogger(__name__)

# --- Tool Input Schema ---
class EvidenceFetcherInput(BaseModel):
    """Input schema for the EvidenceFetcherTool."""
    original_ids: List[str] = Field(description="A list of node or relationship `original_id` values for which to fetch evidence.")
    max_snippet_length: int = Field(default=300, description="Maximum character length for the returned `source_text` snippets.")

class EvidenceFetcherTool(BaseTool):
    """
    Fetches citation metadata (Paper name, year) and source text snippets for a given list of
    node or relationship `original_id`s from the Neo4j graph.
    Useful for providing traceable evidence for claims made based on graph data.
    """
    name: str = "EvidenceFetcher"
    description: str = (
        "Fetch citation details (paper name) and source text snippets for a list of node `original_id`s. "
        "Used to provide evidence for answers derived from the graph."
    )
    args_schema: Type[BaseModel] = EvidenceFetcherInput

    # Neo4j connection
    _neo4j_uri: str = PrivateAttr()
    _neo4j_user: str = PrivateAttr()
    _neo4j_password: str = PrivateAttr()

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, **kwargs):
        """Initialize with Neo4j connection details."""
        super().__init__(**kwargs)
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver is not available.")
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password

    def _run(self, original_ids: List[str], max_snippet_length: int = 300) -> List[Dict[str, Any]]:
        """Execute the evidence fetching Cypher query."""
        if not original_ids:
            logger.warning("EvidenceFetcherTool received an empty list of IDs.")
            return []
        if not NEO4J_AVAILABLE:
            return [{"error": "Neo4j driver not installed. Cannot fetch evidence."}] # Return error within list

        logger.info(f"Fetching evidence for {len(original_ids)} IDs: {original_ids}")

        # Cypher template based on spec
        # Assumes nodes/rels with original_id might have APPEAR_IN relationship to Paper
        # Handles cases where APPEAR_IN or Paper might be missing using OPTIONAL MATCH
        cypher = (
            "UNWIND $ids AS target_id "
            "MATCH (n) WHERE n.original_id = target_id "
            "RETURN n.original_id AS id, n.paper_name AS paper_name, n.source_text AS source_text" # Use RETURN instead of WITH .. RETURN
        )

        driver = None
        try:
            driver = GraphDatabase.driver(self._neo4j_uri, auth=basic_auth(self._neo4j_user, self._neo4j_password))
            driver.verify_connectivity()
            logger.debug(f"Executing Evidence Fetcher Cypher: {cypher} with params: {{ids: {original_ids}}}")

            with driver.session() as session:
                result = session.run(cypher, ids=original_ids)
                evidence_list = []
                for record in result:
                    data = record.data()
                    snippet = data.get('source_text')
                    if snippet and len(snippet) > max_snippet_length:
                        data['source_text'] = snippet[:max_snippet_length] + "..."
                    elif not snippet:
                         data['source_text'] = None # Ensure null if missing

                    # Ensure paper details are null if no paper matched
                    if not data.get('paper_name'):
                        data['paper_name'] = None

                    evidence_list.append(data)

                logger.info(f"Found {len(evidence_list)} evidence records.")
                return evidence_list

        except CypherSyntaxError as e:
             logger.error(f"Cypher syntax error in EvidenceFetcherTool: {e}")
             return [{"error": f"Cypher syntax error: {e}"}]
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Neo4j connection error during evidence fetch: {e}")
            return [{"error": f"Neo4j connection error: {e}"}]
        except Exception as e:
            logger.error(f"An unexpected error occurred during evidence fetch: {e}", exc_info=True)
            return [{"error": f"An unexpected error occurred: {e}"}]
        finally:
            if driver:
                driver.close()

    async def _arun(self, original_ids: List[str], max_snippet_length: int = 300) -> List[Dict[str, Any]]:
        """Asynchronous execution (placeholder)."""
        logger.warning("_arun (async evidence fetch) is not implemented. Falling back to sync.")
        # Requires async neo4j driver
        return self._run(original_ids, max_snippet_length)

# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import os

    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")

    # Example IDs (replace with actual IDs from your graph)
    test_ids = ["CN105582978A_synth_step_32", "US20220410132A1_test_12"]

    if not password:
        logger.info("Error: NEO4J_PASSWORD environment variable not set.")
    elif not NEO4J_AVAILABLE:
        logger.info("Error: neo4j package not installed.")
    else:
        logger.info(f"Testing EvidenceFetcherTool connection to {uri}...")
        try:
            tool = EvidenceFetcherTool(neo4j_uri=uri, neo4j_user=user, neo4j_password=password)

            logger.info(f"\nFetching evidence for IDs: {test_ids}")
            result = tool.run({"original_ids": test_ids, "max_snippet_length": 100})
            logger.info("\nResult:")
            logger.info(json.dumps(result, indent=2, ensure_ascii=False))

        except (ImportError, ConnectionError, RuntimeError) as e:
            logger.info(f"\nTest failed: {e}")
        except Exception as e:
            logger.info(f"\nTest failed with unexpected error: {e}")
            import traceback
            traceback.logger.info_exc() 