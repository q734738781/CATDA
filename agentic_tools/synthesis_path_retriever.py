import logging
import json
from typing import Type, Dict, Any, List, Tuple
import os
from collections import deque # For efficient queue operations

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

# NetworkX Import - REMOVED as iterative approach doesn't strictly need it
# try:
#     import networkx as nx
#     NX_AVAILABLE = True
# except ImportError:
#     nx = None
#     NX_AVAILABLE = False
#     logging.warning("networkx package not found...")
NX_AVAILABLE = False # Explicitly set to False

# Neo4j Imports
try:
    from neo4j import GraphDatabase, basic_auth
    from neo4j.exceptions import ServiceUnavailable, AuthError, CypherSyntaxError
    from neo4j import Result  # Import Result type for example typing
    NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None
    ServiceUnavailable = None
    AuthError = None
    CypherSyntaxError = None
    Result = None # Define Result as None if neo4j is not available
    NEO4J_AVAILABLE = False
    logging.warning("neo4j package not found. SynthesisPathRetrieverTool will not be able to connect. Run `pip install neo4j`")

logger = logging.getLogger(__name__)

# --- Tool Input Schema ---
class SynthesisPathRetrieverInput(BaseModel):
    """Input schema for the SynthesisPathRetrieverTool."""
    target_identifier: str = Field(description="The `name` or `original_id` of the target Catalyst or Chemical node.")
    paper_name: str | None = Field(default=None, description="Optional: The `paper_name` associated with the target node to resolve ambiguity if `target_identifier` is a name.")
    max_depth: int | None = Field(default=20, description="Maximum backward path depth. Defaults to 20.")


class SynthesisPathRetrieverTool(BaseTool):
    """Walk the synthesis DAG backwards iteratively for a target Catalyst or Chemical, returning an ordered list of synthesis steps."""

    name: str = "SynthesisPathRetriever"
    description: str = (
        "Retrieve the synthesis path for a target Catalyst or Chemical (identified by `name` or `original_id`). "
        "Optionally provide the `paper_name` if the `target_identifier` is a name that might exist in multiple papers. "
        "Returns an ordered list of Synthesis node details (method, procedure, etc.) from earliest step to latest step."
    )
    args_schema: Type[BaseModel] = SynthesisPathRetrieverInput

    # Neo4j connection
    _neo4j_uri: str = PrivateAttr()
    _neo4j_user: str = PrivateAttr()
    _neo4j_password: str = PrivateAttr()

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, **kwargs):
        super().__init__(**kwargs)
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver is not available. Please install it using `pip install neo4j`.")
        # Removed NetworkX check
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password

    def _run(self, target_identifier: str, paper_name: str | None = None, max_depth: int = 20) -> List[Dict[str, Any]]:
        if not target_identifier:
            return [{"error": "Input `target_identifier` cannot be empty."}]
        if not NEO4J_AVAILABLE:
            return [{"error": "Neo4j driver not installed."}]

        try:
            depth_limit = int(max_depth)
            if depth_limit <= 0:
                raise ValueError("max_depth must be positive")
        except (ValueError, TypeError):
            logger.warning(f"Invalid max_depth value '{max_depth}', using default of 20.")
            depth_limit = 20

        # Cypher queries
        # Conditionally add paper_name filter
        paper_filter = ""
        params = {"identifier": target_identifier}
        if paper_name:
            paper_filter = "AND n.paper_name = $paper_name "
            params["paper_name"] = paper_name

        find_target_id_query = (
            "MATCH (n) WHERE (n:Catalyst OR n:Chemical) " # Match Catalyst or Chemical
            "AND (n.name = $identifier OR n.original_id = $identifier) "
            f"{paper_filter}" # Add the paper_name filter here
            "RETURN n.original_id AS id "
            "ORDER BY n:Catalyst DESC " # Prioritize Catalyst if name/id matches both
            "LIMIT 1"
        )
        find_producing_synthesis_query = (
            "MATCH (s:Synthesis)-[:SYNTHESIS_OUTPUT]->(target) "
            "WHERE target.original_id = $current_id "
            "RETURN s"
        )
        find_input_chemicals_query = (
            "MATCH (chem:Chemical)-[:SYNTHESIS_INPUT]->(s:Synthesis) "
            "WHERE s.original_id = $current_id "
            "RETURN chem"
        )

        driver = None
        target_node_id = None # Renamed variable
        try:
            driver = GraphDatabase.driver(self._neo4j_uri, auth=basic_auth(self._neo4j_user, self._neo4j_password))
            driver.verify_connectivity()

            # --- Step 1: Resolve target_identifier to original_id ---
            with driver.session() as session:
                try:
                    # Use updated query and input variable name
                    # Pass the updated params dictionary
                    result_target_id = session.execute_read(lambda tx: tx.run(find_target_id_query, params).single())
                    if result_target_id and result_target_id["id"]:
                        target_node_id = result_target_id["id"]
                        # Update log message
                        log_identifier = f"{target_identifier}" + (f" in paper '{paper_name}'" if paper_name else "")
                        logger.info(f"Resolved identifier '{log_identifier}' to target node original_id: '{target_node_id}'")
                    else:
                        # Update log message
                        log_identifier = f"{target_identifier}" + (f" in paper '{paper_name}'" if paper_name else "")
                        logger.error(f"Could not find a unique Catalyst or Chemical node with identifier: '{log_identifier}'")
                        return [{"error": f"Target node '{log_identifier}' not found."}]
                except Exception as e:
                     # Update log message
                     log_identifier = f"{target_identifier}" + (f" in paper '{paper_name}'" if paper_name else "")
                     logger.error(f"Error resolving target identifier '{log_identifier}': {e}", exc_info=True)
                     return [{"error": f"Error finding target node: {e}"}]

            # --- Step 2: Iterative BFS starting from the resolved original_id ---
            queue = deque([(target_node_id, 0)])
            visited = {target_node_id}
            synthesis_steps_data = {} # Store synthesis steps found

            with driver.session() as session:
                while queue:
                    current_id, current_dist = queue.popleft()

                    if current_dist >= depth_limit:
                        logger.debug(f"Reached depth limit ({depth_limit}) at node {current_id}. Stopping exploration from here.")
                        continue

                    # Find Synthesis step(s) producing the current node
                    try:
                        result_synthesis = session.execute_read(lambda tx: list(tx.run(find_producing_synthesis_query, current_id=current_id)))
                        for record in result_synthesis:
                            s_node = record["s"]
                            s_id = s_node.get("original_id")
                            if s_id and s_id not in visited:
                                visited.add(s_id)
                                s_dist = current_dist + 1
                                synthesis_steps_data[s_id] = {
                                    "properties": {
                                        "original_id": s_id,
                                        "method": s_node.get("method"),
                                        "procedure": s_node.get("procedure"),
                                        "name": s_node.get("name"),
                                        "paper_name": s_node.get("paper_name")
                                    },
                                    "distance": s_dist
                                }
                                queue.append((s_id, s_dist))
                                logger.debug(f"Found Synthesis: {s_id} (dist={s_dist}) via output from {current_id}")

                    except CypherSyntaxError as e:
                         logger.error(f"Cypher error finding producing synthesis for {current_id}: {e}")
                         return [{"error": f"Cypher syntax error: {e}"}]
                    except Exception as e:
                         logger.error(f"Error finding producing synthesis for {current_id}: {e}", exc_info=True)

                    # Find Chemical(s) input to the current node (if it's a Synthesis node)
                    if current_id in synthesis_steps_data:
                        try:
                            result_chemicals = session.execute_read(lambda tx: list(tx.run(find_input_chemicals_query, current_id=current_id)))
                            for record in result_chemicals:
                                chem_node = record["chem"]
                                chem_id = chem_node.get("original_id")
                                if chem_id and chem_id not in visited:
                                    visited.add(chem_id)
                                    chem_dist = current_dist + 1
                                    queue.append((chem_id, chem_dist))
                                    logger.debug(f"Found Chemical input: {chem_id} (dist={chem_dist}) for Synthesis {current_id}")
                        except CypherSyntaxError as e:
                            logger.error(f"Cypher error finding input chemicals for {current_id}: {e}")
                            return [{"error": f"Cypher syntax error: {e}"}]
                        except Exception as e:
                             logger.error(f"Error finding input chemicals for {current_id}: {e}", exc_info=True)

            # --- Step 3: Post-processing ---
            final_steps = list(synthesis_steps_data.values())
            final_steps.sort(key=lambda x: x.get('distance', -1), reverse=True)
            output_steps = [step['properties'] for step in final_steps]

            # Update log message
            log_identifier = f"{target_identifier}" + (f" in paper '{paper_name}'" if paper_name else "")
            logger.info(f"Iteratively fetched and ordered {len(output_steps)} synthesis steps for target identifier '{log_identifier}'.")
            return output_steps

        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Neo4j connection error during synthesis path fetch: {e}")
            return [{"error": f"Neo4j connection error: {e}"}]
        except Exception as e:
            logger.error(f"Unexpected error during iterative synthesis path exploration: {e}", exc_info=True)
            return [{"error": f"Unexpected error: {e}"}]
        finally:
            if driver:
                driver.close()

    async def _arun(self, target_identifier: str, paper_name: str | None = None, max_depth: int = 20) -> List[Dict[str, Any]]:
        logger.warning("_arun (async synthesis path retriever) not implemented, falling back to sync.")
        # Update call to use renamed variable and include paper_name
        return self._run(target_identifier, paper_name=paper_name, max_depth=max_depth)

# --- Example Usage ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load Neo4j credentials from environment variables
    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")

    # Update example variables
    test_id = "US20180345257A1_chem_Mo_ZSM5_5C" # Hypothetical Catalyst ID
    test_name = "催化剂 C-13" # Hypothetical Chemical Name
    test_paper = "Hypothetical_Paper_A.pdf" # Example paper name

    if not password:
        logger.error("NEO4J_PASSWORD environment variable not set. Cannot run example.")
    elif not NEO4J_AVAILABLE:
        logger.error("neo4j package not installed. Cannot run example.")
    # Removed elif for NX_AVAILABLE
    else:
        logger.info(f"Testing SynthesisPathRetrieverTool connection to {uri}...")
        try:
            # Removed NX check before init
            tool = SynthesisPathRetrieverTool(neo4j_uri=uri, neo4j_user=user, neo4j_password=password)

            # Test with ID
            logger.info(f"Fetching synthesis path for ID: {test_id}")
            # Use renamed input field
            result_catalyst = tool.run({"target_identifier": test_id})
            logger.info("Result (Catalyst ID):")
            logger.info(json.dumps(result_catalyst, indent=2, ensure_ascii=False))

            # Test with Name
            logger.info(f"Fetching synthesis path for CHEMICAL NAME: {test_name}")
            # Use renamed input field
            result_chem_name = tool.run({"target_identifier": test_name, "max_depth": 10}) # Example smaller depth
            logger.info("Result (Chemical Name):")
            logger.info(json.dumps(result_chem_name, indent=2, ensure_ascii=False))

            # Test with Name and Paper Name
            logger.info(f"Fetching synthesis path for CHEMICAL NAME: {test_name} in PAPER: {test_paper}")
            # Use renamed input field and new paper_name field
            result_chem_name_paper = tool.run({"target_identifier": test_name, "paper_name": test_paper, "max_depth": 10})
            logger.info("Result (Chemical Name with Paper):")
            logger.info(json.dumps(result_chem_name_paper, indent=2, ensure_ascii=False))


        except ImportError as e:
             logger.error(f"Import error during tool initialization or run: {e}")
        except (ServiceUnavailable, AuthError) as e:
             logger.error(f"Neo4j connection failed during example run: {e}")
        except Exception as e:
            logger.error(f"Example run failed with unexpected error: {e}", exc_info=True)