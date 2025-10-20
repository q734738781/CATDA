import logging
import os
import json
import time
from typing import Type, Dict, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

# Try to import neo4j and handle potential ImportError
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
    logging.warning("neo4j package not found. GraphSchemaTool will not be able to connect. Run `pip install neo4j`")

logger = logging.getLogger(__name__)

class GraphSchemaInput(BaseModel):
    """Input schema for the GraphSchemaTool. Optionally accepts node_type and prefix for filtering."""
    node_type: str | None = Field(
        default=None,
        description="Optional node label to filter the schema by. Returns only schema for this node type."
    )
    prefix: str | None = Field(
        default=None,
        description="Optional prefix to filter node properties by. Only properties starting with this prefix will be returned."
    )

    # Add a configuration to allow extra fields if needed, though ideally not used by agent
    # class Config:
    #     extra = "allow"

class GraphSchemaTool(BaseTool):
    """
    Tool to retrieve the schema of the Neo4j graph.
    Provides information about node labels, relationship types, and property keys.
    Caches the schema for a short duration to improve performance.
    """
    name: str = "GraphSchema"
    description: str = (
        "Retrieve the full graph schema: node labels, relationship types, and all property keys. "
        "Can optionally filter fields by node type or prefix. That will return only the fields for corresponding nodes or/and with specified prefix"
        "Call when unsure about available graph structure or property names."
    )
    args_schema: Type[BaseModel] = GraphSchemaInput

    _neo4j_uri: str = PrivateAttr()
    _neo4j_user: str = PrivateAttr()
    _neo4j_password: str = PrivateAttr()
    _cache_ttl_seconds: int = PrivateAttr(default=3600) # Cache for 1 hour by default
    _last_fetch_time: float = PrivateAttr(default=0.0)
    _cached_schema: Dict[str, Any] | None = PrivateAttr(default=None)


    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, cache_ttl_seconds: int = 3600, cache_dir: str | None = None, **kwargs):
        """Initialize the tool with Neo4j connection details and cache TTL."""
        super().__init__(**kwargs)
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver is not available. Please install it using `pip install neo4j`.")
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._cache_ttl_seconds = cache_ttl_seconds

        # LMDB persistent cache setup
        try:
            import lmdb
            self._lmdb_env = None
            if cache_dir:
                cache_path = os.path.join(cache_dir, 'graph_schema')
                os.makedirs(cache_path, exist_ok=True)
                self._lmdb_env = lmdb.open(
                    cache_path,
                    map_size=1 << 29,  # 512 MiB default
                    subdir=True,
                    max_dbs=1,
                    readonly=False,
                    lock=False,
                )
        except ImportError:
            self._lmdb_env = None
            logger.warning("lmdb not installed – GraphSchemaTool will run without persistent caching.")

    # Internal method to fetch schema from DB (no longer decorated with lru_cache)
    def _fetch_schema_from_db(self, uri: str, user: str, password: str) -> Dict[str, Any]:
        """Internal method to fetch schema using apoc.meta.schema."""
        logger.info(f"Fetching schema via apoc.meta.schema from Neo4j: {uri}")
        driver = None
        schema_info = {"labels": {}, "relationships": set(), "properties": set()}

        try:
            driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
            driver.verify_connectivity()

            with driver.session() as session:
                 try:
                     meta_result = session.run("CALL apoc.meta.schema()")
                     meta_data = meta_result.single()[0] # APOC returns a nested dict

                     for label, data in meta_data.items():
                         if data['type'] == 'node':
                             props = sorted(list(data.get('properties', {}).keys())) # Sort properties here
                             schema_info["labels"][label] = props
                             schema_info["properties"].update(props)
                         elif data['type'] == 'relationship':
                             schema_info["relationships"].add(label) # APOC uses rel type as key
                             # Optionally extract relationship properties if needed
                             rel_props = sorted(list(data.get('properties', {}).keys()))
                             schema_info["properties"].update(rel_props) # Add rel props if desired

                     schema_info["relationships"] = sorted(list(schema_info["relationships"]))
                     logger.info("Successfully fetched schema using apoc.meta.schema()")

                 except (CypherSyntaxError, IndexError, TypeError, Exception) as e:
                     # Catch potential errors if apoc is not installed or returns unexpected format
                     logger.error(f"apoc.meta.schema() failed: {e}. Cannot retrieve schema.", exc_info=True)
                     raise RuntimeError("Failed to fetch schema using apoc.meta.schema.") from e

            # Convert property set to sorted list for consistent output
            schema_info["properties"] = sorted(list(schema_info["properties"]))
            return schema_info

        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Neo4j connection error during schema fetch: {e}")
            raise ConnectionError(f"Could not connect to Neo4j database for schema fetch: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during schema fetch: {e}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred during schema fetch: {e}") from e
        finally:
            if driver:
                driver.close()

    def _run(self, node_type: str | None = None, prefix: str | None = None, **kwargs) -> Dict[str, Any]:
        """Execute the schema fetch, utilizing cache and applying filters."""
        current_time = time.time()

        # node_type and prefix are now directly available as arguments

        # Use a consistent cache key based only on connection details, not filters
        cache_key_dict = {"uri": self._neo4j_uri, "user": self._neo4j_user}
        cache_key = json.dumps(cache_key_dict, sort_keys=True).encode()

        full_schema: Dict[str, Any] | None = None

        # Attempt persistent cache read first
        if getattr(self, "_lmdb_env", None):
            with self._lmdb_env.begin(write=False) as txn:
                cached_bytes = txn.get(cache_key)
                if cached_bytes:
                    try:
                        cached_obj = json.loads(cached_bytes.decode())
                        ts = cached_obj.get("ts", 0)
                        if current_time - ts < self._cache_ttl_seconds:
                            logger.info("Retrieved full schema from LMDB persistent cache.")
                            full_schema = cached_obj.get("schema", {})
                        else:
                            logger.info("LMDB persistent cache entry expired.")
                    except Exception:
                        logger.warning("Corrupted LMDB entry for GraphSchemaTool – fetching fresh schema.")

        # Check in-memory cache if not found in persistent cache
        if full_schema is None and self._cached_schema and (current_time - self._last_fetch_time < self._cache_ttl_seconds):
            logger.info("Retrieved full schema from in-memory cache.")
            full_schema = self._cached_schema

        # Fetch from DB if not in any cache or cache expired
        if full_schema is None:
            logger.info("Cache empty or expired, fetching fresh schema from DB.")
            try:
                full_schema = self._fetch_schema_from_db(self._neo4j_uri, self._neo4j_user, self._neo4j_password)
                self._cached_schema = full_schema # Update in-memory cache
                self._last_fetch_time = current_time

                # Persist to LMDB
                if getattr(self, "_lmdb_env", None):
                    try:
                        payload = json.dumps({"ts": current_time, "schema": full_schema}).encode()
                        with self._lmdb_env.begin(write=True) as txn:
                            txn.put(cache_key, payload)
                        logger.info("Persisted fresh schema to LMDB cache.")
                    except Exception as e:
                        logger.warning(f"Failed to write GraphSchemaTool schema to LMDB cache: {e}")

            except (ConnectionError, RuntimeError) as e:
                return {"error": f"Failed to fetch schema: {e}"}
            except Exception as e:
                logger.error(f"Unexpected error during schema fetch in _run: {e}", exc_info=True)
                return {"error": f"An unexpected error occurred during schema fetch: {e}"}

        # --- Apply Filtering ---
        if node_type or prefix:
            logger.info(f"Applying filters: node_type='{node_type}', prefix='{prefix}'")
            filtered_schema: Dict[str, Any] = {"labels": {}}

            if full_schema and "labels" in full_schema:
                for label, properties in full_schema["labels"].items():
                    # 1. Filter by node_type
                    if node_type and label != node_type:
                        continue # Skip this label if node_type is specified and doesn't match

                    # 2. Filter by prefix
                    if prefix:
                        filtered_properties = [prop for prop in properties if prop.startswith(prefix)]
                        # Only add label if it has properties matching the prefix
                        if filtered_properties:
                            filtered_schema["labels"][label] = filtered_properties
                    else:
                         # No prefix filter, include all properties for the matched label
                         filtered_schema["labels"][label] = properties

            logger.debug(f"Filtered schema: {filtered_schema}")
            return filtered_schema # Return only filtered labels
        else:
            # No filters applied, return the full schema
            logger.info("No filters applied, returning full schema.")
            return full_schema if full_schema else {} # Return full schema or empty dict if fetch failed

    async def _arun(self, node_type: str | None = None, prefix: str | None = None, **kwargs) -> Dict[str, Any]:
        """Asynchronous execution (placeholder)."""
        logger.warning("_arun (async schema fetch) is not implemented. Falling back to sync.")
        # Pass along the filter arguments to the sync version
        return self._run(node_type=node_type, prefix=prefix, **kwargs)

# Example usage (for testing)
if __name__ == '__main__':
    import os
    import json
    logging.basicConfig(level=logging.INFO)

    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")

    if not password:
        logger.info("Error: NEO4J_PASSWORD environment variable not set.")
    elif not NEO4J_AVAILABLE:
        logger.info("Error: neo4j package not installed.")
    else:
        logger.info(f"Testing GraphSchemaTool connection to {uri}...")
        try:
            tool = GraphSchemaTool(neo4j_uri=uri, neo4j_user=user, neo4j_password=password)
            schema_result = tool.run({}) # Pass empty dict or let default handle it
            logger.info("\nSchema Result:")
            logger.info(json.dumps(schema_result, indent=2, ensure_ascii=False))

            # Test caching
            logger.info("\nFetching schema again (should use cache)...")
            start_time = time.time()
            schema_result_cached = tool.run({})
            end_time = time.time()
            logger.info(f"Second fetch time: {end_time - start_time:.4f} seconds")
            # logger.info(json.dumps(schema_result_cached, indent=2)) # Optionally logger.info again

        except (ImportError, ConnectionError, RuntimeError) as e:
            logger.info(f"\nTest failed: {e}")
        except Exception as e:
            logger.info(f"\nTest failed with unexpected error: {e}")
            import traceback
            traceback.logger.info_exc()

        # --- New Filter Tests ---
        if 'tool' in locals() and NEO4J_AVAILABLE: # Check if tool was initialized
            logger.info("\n--- Testing Filters ---")

            # Test Case 1: No filters (same as original test)
            logger.info("\n1. Fetching schema with NO filters...")
            start_time = time.time()
            schema_result_no_filter = tool.run({}) # Empty dict for no filters
            end_time = time.time()
            logger.info(f"Fetch time: {end_time - start_time:.4f} seconds (should be cached)")
            logger.info(json.dumps(schema_result_no_filter, indent=2, ensure_ascii=False))


            # Test Case 2: Filter by node_type
            test_node_type = "Catalyst" # Replace with an actual node type from your graph if needed
            logger.info(f"\n2. Fetching schema with node_type='{test_node_type}'...")
            start_time = time.time()
            schema_result_node_type = tool.run({"node_type": test_node_type})
            end_time = time.time()
            logger.info(f"Fetch time: {end_time - start_time:.4f} seconds")
            logger.info(json.dumps(schema_result_node_type, indent=2, ensure_ascii=False))


            # Test Case 3: Filter by node_type and prefix
            test_prefix = "composition" # Replace with an actual prefix if needed
            logger.info(f"\n3. Fetching schema with node_type='{test_node_type}' and prefix='{test_prefix}'...")
            start_time = time.time()
            schema_result_node_prefix = tool.run({"node_type": test_node_type, "prefix": test_prefix})
            end_time = time.time()
            logger.info(f"Fetch time: {end_time - start_time:.4f} seconds")
            logger.info(json.dumps(schema_result_node_prefix, indent=2, ensure_ascii=False))


            # Test Case 4: Filter by prefix only
            logger.info(f"\n4. Fetching schema with prefix='{test_prefix}' only...")
            start_time = time.time()
            schema_result_prefix_only = tool.run({"prefix": test_prefix})
            end_time = time.time()
            logger.info(f"Fetch time: {end_time - start_time:.4f} seconds")
            logger.info(json.dumps(schema_result_prefix_only, indent=2, ensure_ascii=False))

            # Test Case 5: Filter by non-existent node_type
            logger.info("\n5. Fetching schema with non-existent node_type='NonExistentType'...")
            start_time = time.time()
            schema_result_non_existent = tool.run({"node_type": "NonExistentType"})
            end_time = time.time()
            logger.info(f"Fetch time: {end_time - start_time:.4f} seconds")
            logger.info(json.dumps(schema_result_non_existent, indent=2, ensure_ascii=False)) # Should show empty labels

        else:
             logger.info("\nSkipping filter tests due to initialization failure or missing neo4j package.") 