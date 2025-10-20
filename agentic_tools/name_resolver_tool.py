import logging
import json
import re
from typing import Type, List, Dict, Any, Tuple
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import os

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
    logging.warning("neo4j package not found. NameResolverTool will not be able to connect. Run `pip install neo4j`")

# Vector Search Imports (Optional - requires faiss-cpu, sentence-transformers)
try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    raise RuntimeError("faiss-cpu or sentence-transformers not found. Vector search in NameResolverTool is disabled. Run `pip install faiss-cpu sentence-transformers`")


logger = logging.getLogger(__name__)

# --- Tool Input Schema ---
class NameResolverInput(BaseModel):
    """Input schema for the NameResolverTool."""
    free_name: str = Field(description="The user-provided name (potentially alias or typo) for a catalyst, chemical, or material.")
    target_labels: List[str] | None = Field(default=None, description="Optional list of node labels (e.g., ['Catalyst', 'Chemical']) to restrict the search.")
    top_k: int = Field(default=5, description="Maximum number of candidates to return.")

class NameResolverTool(BaseTool):
    """
    Resolves potentially ambiguous or misspelled names for entities (Catalyst, Chemical, BasicMaterial)
    against the knowledge graph using semantic (vector) search.
    Returns the best guess and a list of ranked candidates based on semantic similarity.
    Requires Neo4j connection to fetch node details after vector search.
    """
    name: str = "NameResolver"
    description: str = (
        "Resolve ambiguous or misspelled Catalyst, Chemical, or BasicMaterial names provided by the user. "
        "Uses vector search to map the input name to canonical graph identifiers (`name`, `original_id`). Returns best match and candidates."
    )
    args_schema: Type[BaseModel] = NameResolverInput

    # Neo4j connection
    _neo4j_uri: str = PrivateAttr()
    _neo4j_user: str = PrivateAttr()
    _neo4j_password: str = PrivateAttr()

    # Vector search config
    _vector_embedding_model: str = PrivateAttr(default='all-MiniLM-L6-v2')
    _vector_model_instance: Any = PrivateAttr(default=None) # SentenceTransformer model
    _vector_index_instance: Any = PrivateAttr(default=None) # FAISS index
    _vector_id_map: List[str] | None = PrivateAttr(default=None) # Maps FAISS index position to node ID/name
    _node_labels_for_index: List[str] | None = PrivateAttr(default=None) # Labels used when BUILDING the index
    _cosine_similarity_threshold: float = PrivateAttr(default=0.5) # Threshold for vector results

    # Determine cache paths for vector index and id map
    _vector_index_path: str | None = PrivateAttr(default=None)
    _vector_id_map_path: str | None = PrivateAttr(default=None)

    # User-provided regex mapping (optional)
    _regex_map_path: str | None = PrivateAttr(default=None)
    _user_regex_patterns: List[Tuple[re.Pattern, str]] | None = PrivateAttr(default=None)
    _regex_map_version: float | None = PrivateAttr(default=None)

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        node_labels_for_index: List[str] | None = None, # Labels used for index build
        vector_embedding_model: str = 'all-MiniLM-L6-v2',
        cosine_similarity_threshold: float = 0.5, # Similarity threshold
        cache_dir: str | None = None,
        regex_map_path: str | None = None,
        **kwargs
    ):
        """Initialize with connection details and vector search parameters."""
        super().__init__(**kwargs)
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver is not available.")
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._vector_embedding_model = vector_embedding_model
        self._node_labels_for_index = node_labels_for_index # Store labels used for index build
        self._cosine_similarity_threshold = cosine_similarity_threshold
        self._regex_map_path = regex_map_path

        # Determine cache paths for vector index and id map
        self._vector_index_path = None
        self._vector_id_map_path = None
        if cache_dir:
            label_suffix = "all" if not node_labels_for_index else "_".join(sorted([lbl.lower() for lbl in node_labels_for_index]))
            vec_dir = os.path.join(cache_dir, "name_resolver_vectors")
            os.makedirs(vec_dir, exist_ok=True)
            self._vector_index_path = os.path.join(vec_dir, f"nr_{label_suffix}.faiss")
            self._vector_id_map_path = os.path.join(vec_dir, f"nr_{label_suffix}_id_map.json")
        else:
            logger.warning("Cache directory not specified for NameResolverTool. Vector index/cache will not be persisted.")

        # Load optional user regex mapping
        self._load_user_regex_mapping()

        # Attempt to load or (re)build vector search structures
        # This method now only builds the index based on specified labels
        logger.info("Initializing NameResolverTool vector search index... \n If this is the first time you are using this tool or graph has changed, it may take a while.")
        self._initialize_vector_search()
        logger.info("NameResolverTool vector search index initialized.")

        # Persistent LMDB cache setup
        try:
            import lmdb  # Local import to avoid mandatory dependency
            self._lmdb_env = None
            if cache_dir:
                cache_path = os.path.join(cache_dir, 'name_resolver_cache') # Separate cache name
                os.makedirs(cache_path, exist_ok=True)
                # 1 GiB default map size (adjust as needed)
                self._lmdb_env = lmdb.open(
                    cache_path,
                    map_size=1 << 30,
                    subdir=True,
                    max_dbs=1,
                    readonly=False,
                    lock=False,
                )
        except ImportError:
            self._lmdb_env = None
            logger.warning("lmdb not installed – NameResolverTool will run without persistent caching.")

    def _load_user_regex_mapping(self) -> None:
        """Load and compile user-provided regex->canonical name mappings from JSON file (optional)."""
        self._user_regex_patterns = []
        self._regex_map_version = None
        if not self._regex_map_path:
            return
        try:
            if not os.path.exists(self._regex_map_path):
                logger.warning(f"Regex mapping file not found: {self._regex_map_path}. Skipping regex mapping.")
                return
            with open(self._regex_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            compiled: List[Tuple[re.Pattern, str]] = []
            # Support either dict {pattern: name} or list of {"pattern":..., "name":...}
            if isinstance(data, dict):
                items = data.items()
            elif isinstance(data, list):
                items = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    pat = item.get("pattern") or item.get("regex")
                    name = item.get("name") or item.get("canonical")
                    if pat and name:
                        items.append((pat, name))
            else:
                logger.warning("Unsupported JSON format for regex mapping. Expect dict or list of objects.")
                return

            for entry in items:
                if isinstance(entry, tuple):
                    pattern_str, canonical_name = entry
                else:
                    pattern_str, canonical_name = entry[0], entry[1]
                try:
                    compiled.append((re.compile(pattern_str, re.IGNORECASE), str(canonical_name)))
                except re.error as rex:
                    logger.warning(f"Invalid regex pattern skipped in NameResolverTool: {pattern_str!r} ({rex})")
            self._user_regex_patterns = compiled
            try:
                self._regex_map_version = os.path.getmtime(self._regex_map_path)
            except Exception:
                self._regex_map_version = None
            logger.info(f"Loaded {len(self._user_regex_patterns)} user regex mappings for NameResolverTool.")
        except Exception as e:
            logger.warning(f"Failed to load regex mapping file for NameResolverTool: {e}")

    def _initialize_vector_search(self):
        """Load vector search assets if they exist, otherwise build them from Neo4j for the specified labels."""
        if not VECTOR_SEARCH_AVAILABLE:
            logger.warning("Vector search libraries missing – running without semantic search.")
            return

        current_names: List[str] = []
        driver = None
        try:
            self._vector_model_instance = SentenceTransformer(self._vector_embedding_model) # Load model first

            # Construct WHERE clause parts
            where_clauses = []
            params = {}
            if self._node_labels_for_index:
                # Check if node has AT LEAST ONE of the specified labels
                where_clauses.append("size([lbl IN labels(n) WHERE lbl IN $labels | 1]) > 0")
                params["labels"] = self._node_labels_for_index
            # Always ensure the identifier property exists
            # IMPORTANT: Assuming 'name' is the identifier stored in the vector_id_map
            identifier_property = "name"
            where_clauses.append(f"n.{identifier_property} IS NOT NULL")

            # Combine WHERE clauses
            where_clause = ""
            if where_clauses:
                where_clause = "WHERE " + " AND ".join(where_clauses)

            # Final Query
            query = (
                f"MATCH (n) {where_clause} "
                f"RETURN DISTINCT n.{identifier_property} AS identifier ORDER BY identifier"
            )

            logger.debug(f"Executing Initial Name Fetch Cypher: {query} with params {params}")

            driver = GraphDatabase.driver(self._neo4j_uri, auth=basic_auth(self._neo4j_user, self._neo4j_password))
            with driver.session() as session:
                result = session.run(query, **params)
                current_names = sorted([rec["identifier"] for rec in result if rec["identifier"]]) # Unique & sorted
            logger.debug(f"Fetched {len(current_names)} unique identifiers ({identifier_property}s) from Neo4j for vector index build.")
            logger.info(f"INFO: NameResolverTool search space initialized with {len(current_names)} unique names for vector index.")
        except (ServiceUnavailable, AuthError, CypherSyntaxError, Exception) as e:
            logger.error(f"Failed to fetch current names/load model for vector index check: {e}", exc_info=True)
            self._vector_model_instance = None # Ensure model isn't partially loaded on error
            return # Cannot proceed without current names or model
        finally:
            if driver:
                driver.close()

        if not current_names:
             logger.warning("No names found in Neo4j matching criteria. Skipping vector index build.")
             return

        # --- 2. Check Cached Names ---
        cached_names: List[str] | None = None
        cache_valid = False
        if self._vector_id_map_path and os.path.exists(self._vector_id_map_path) and self._vector_index_path and os.path.exists(self._vector_index_path):
            try:
                with open(self._vector_id_map_path, "r", encoding="utf-8") as f:
                    cached_names = json.load(f)
                # Simple equality check might be okay if order is guaranteed (by sorting)
                if current_names == cached_names:
                    cache_valid = True
                    logger.info(f"Cached name list ({self._vector_id_map_path}) matches current Neo4j names.")
                else:
                    logger.info("Neo4j names have changed or cache mismatch. Rebuilding NameResolverTool vector index.")
            except Exception as e:
                logger.warning(f"Could not read/validate cached name ID map ({self._vector_id_map_path}): {e}. Rebuilding index.")

        # --- 3. Load or Rebuild ---
        if cache_valid and cached_names:
            try:
                logger.info(f"Loading cached FAISS index ({self._vector_index_path}) for NameResolverTool.")
                self._vector_index_instance = faiss.read_index(self._vector_index_path)
                self._vector_id_map = cached_names # Use the names read from the valid cache
                logger.info(f"Successfully loaded {len(self._vector_id_map)} name vectors from cache.")
                logger.info(f"INFO: NameResolverTool successfully loaded index and map from cache.")
            except Exception as e:
                logger.error(f"Failed to load cached FAISS index ({self._vector_index_path}): {e}. Attempting rebuild.")
                logger.info(f"ERROR: NameResolverTool failed to load cached index/map: {e}. Forcing rebuild.")
                cache_valid = False # Force rebuild

        if not cache_valid:
            logger.info("Building NameResolverTool vector index from Neo4j names...")
            logger.info("INFO: NameResolverTool attempting to build new vector index...")
            if not self._vector_model_instance: # Should be loaded already, but double check
                 logger.error("Embedding model not loaded. Cannot build vector index.")
                 logger.info("ERROR: NameResolverTool embedding model not loaded, cannot build index.")
                 return
            try:
                # Encode the fetched names (identifiers)
                logger.info("Encoding names for vector index...")
                embeddings = self._vector_model_instance.encode(current_names, show_progress_bar=False, normalize_embeddings=True).astype("float32")
                logger.info(f"Generated embeddings of shape: {embeddings.shape}")
                if embeddings.ndim == 1: # Handle case of single embedding
                     embeddings = embeddings.reshape(1, -1)
                if embeddings.shape[0] == 0:
                    logger.warning("No embeddings generated, cannot build index.")
                    logger.info("ERROR: NameResolverTool generated 0 embeddings, cannot build index.")
                    return

                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim) # Using Inner Product (cosine similarity on normalized vectors)
                logger.info(f"Adding {embeddings.shape[0]} embeddings to FAISS index...")
                index.add(embeddings)
                logger.info(f"FAISS index built successfully.")

                self._vector_index_instance = index
                self._vector_id_map = current_names # Use the fresh names from Neo4j
                logger.info(f"INFO: NameResolverTool successfully built index and map in memory.")

                # Persist the new index and map
                if self._vector_index_path and self._vector_id_map_path:
                    logger.info(f"Attempting to save index to {self._vector_index_path} and map to {self._vector_id_map_path}")
                    faiss.write_index(index, self._vector_index_path)
                    with open(self._vector_id_map_path, "w", encoding="utf-8") as f:
                        json.dump(current_names, f, indent=2) # Save pretty-logger.infoed JSON
                    logger.info(f"Built and saved new name vector index ({len(current_names)} names) to {self._vector_index_path}.")
                    logger.info(f"INFO: NameResolverTool successfully saved new index and map.")
                else:
                    logger.warning("Cache directory not configured. Vector index will not be persisted.")

            except Exception as e:
                logger.error(f"Failed during vector index build/save: {e}", exc_info=True)
                logger.info(f"ERROR: NameResolverTool failed during vector index build/save: {e}")
                self._vector_index_instance = None
                self._vector_id_map = None

    # Persistent cache using LMDB (if available). Falls back to normal execution when unavailable.
    #@functools.lru_cache() # In-memory cache using lru_cache
    def _perform_search(self, name: str, labels: tuple | None = None, k: int = 5) -> Dict[str, Any]:
        """Internal search logic using vector search and subsequent DB lookup, cached."""
        cache_key_dict = {"name": name, "labels": labels, "k": k, "regex_ver": self._regex_map_version}
        cache_key = json.dumps(cache_key_dict, sort_keys=True).encode()

        # 1. Try to read from LMDB if available
        if getattr(self, "_lmdb_env", None):
            with self._lmdb_env.begin(write=False) as txn:
                cached_bytes = txn.get(cache_key)
                if cached_bytes is not None:
                    try:
                        cached_result = json.loads(cached_bytes.decode())
                        logger.info(f"Returning cached result for NameResolverTool: {cache_key_dict}")
                        return cached_result
                    except Exception as e:
                        # Corrupted cache entry – ignore and recompute
                        logger.warning(f"Corrupted LMDB cache entry for NameResolverTool ({e}) – recomputing result.")

        # 0. Regex mapping (higher priority than vector search)
        if self._user_regex_patterns:
            matched_canonicals: List[str] = []
            for pattern, canonical in self._user_regex_patterns:
                try:
                    if pattern.search(name):
                        matched_canonicals.append(canonical)
                except Exception:
                    continue
            if matched_canonicals:
                # Fetch details for matched canonical names from Neo4j
                detailed_candidates = {}
                driver = None
                try:
                    driver = GraphDatabase.driver(self._neo4j_uri, auth=basic_auth(self._neo4j_user, self._neo4j_password))
                    with driver.session() as session:
                        where_clauses = ["n.name IN $identifiers"]
                        params = {"identifiers": list(dict.fromkeys(matched_canonicals))}
                        if labels:
                            label_match = " OR ".join([f"lbl = '{lbl}'" for lbl in labels])
                            where_clauses.append(f"size([lbl IN labels(n) WHERE {label_match} | 1]) > 0")
                        where_clause = "WHERE " + " AND ".join(where_clauses)
                        cypher = (
                            f"MATCH (n) {where_clause} "
                            f"RETURN n.name AS name, n.original_id AS original_id, labels(n) AS node_labels"
                        )
                        result = session.run(cypher, params)
                        for record in result:
                            if record['original_id'] is None:
                                continue
                            detailed_candidates[record['name']] = {
                                "name": record['name'],
                                "original_id": record['original_id'],
                                "labels": record['node_labels'],
                                "score": 1.0,
                                "source": "regex"
                            }
                except (ServiceUnavailable, AuthError, CypherSyntaxError, Exception) as e:
                    logger.error(f"Neo4j error during node detail fetch for regex mapping: {e}")
                    detailed_candidates = {}
                finally:
                    if driver:
                        driver.close()

                sorted_candidates = list(detailed_candidates.values())[:k]
                final_result = {
                    "best_match": sorted_candidates[0] if sorted_candidates else None,
                    "candidates": sorted_candidates
                }
                if getattr(self, "_lmdb_env", None):
                    try:
                        with self._lmdb_env.begin(write=True) as txn:
                            txn.put(cache_key, json.dumps(final_result).encode())
                    except Exception as e:
                        logger.warning(f"Failed to write NameResolverTool result to LMDB cache: {e}")
                return final_result

        logger.info(f"Performing vector-based name resolution for: '{name}', labels: {labels}, k: {k}")
        vector_candidates = {}
        candidate_identifiers = []

        # 1. Vector Search (if available and initialized)
        if self._vector_model_instance and self._vector_index_instance and self._vector_id_map:
            try:
                query_embedding = self._vector_model_instance.encode([name.lower()], normalize_embeddings=True).astype("float32")
                if query_embedding.ndim == 1: # Handle single query embedding
                    query_embedding = query_embedding.reshape(1, -1)

                # Search more initially to allow for filtering/thresholding
                distances, indices = self._vector_index_instance.search(query_embedding, k * 3)

                for i, idx in enumerate(indices[0]):
                    if idx < 0 or idx >= len(self._vector_id_map): continue # Skip invalid indices

                    # Assuming inner product index (IndexFlatIP) and normalized embeddings,
                    # distance is cosine similarity
                    similarity_score = float(distances[0][i])

                    if similarity_score >= self._cosine_similarity_threshold:
                        identifier = self._vector_id_map[idx] # Get the identifier (e.g., name) from map
                        if identifier not in vector_candidates: # Store highest score per identifier
                             candidate_identifiers.append(identifier)
                             vector_candidates[identifier] = similarity_score
                        # else: Keep the first (highest score) found if duplicates occur
                    else:
                        # Stop adding candidates if score drops below threshold (since results are ordered)
                        break
                logger.debug(f"Vector search found {len(candidate_identifiers)} candidates above threshold: {candidate_identifiers}")

            except Exception as e:
                logger.error(f"Error during vector search: {e}", exc_info=True)
                # Don't fail the whole tool
        elif self._vector_index_path:
             logger.warning("Vector search components specified but not loaded/initialized correctly. Skipping vector search.")

        # 2. Fetch Node Details from Neo4j for Vector Candidates
        detailed_candidates = {}
        if candidate_identifiers:
            driver = None
            try:
                driver = GraphDatabase.driver(self._neo4j_uri, auth=basic_auth(self._neo4j_user, self._neo4j_password))
                with driver.session() as session:
                    # Construct the label filter part of the query if provided
                    where_clauses = ["n.name IN $identifiers"]
                    params = {"identifiers": candidate_identifiers}
                    if labels:
                        # Match nodes that have AT LEAST ONE of the target labels
                        label_match = " OR ".join([f"lbl = '{lbl}'" for lbl in labels])
                        where_clauses.append(f"size([lbl IN labels(n) WHERE {label_match} | 1]) > 0")
                        # No need to add labels to params here

                    where_clause = "WHERE " + " AND ".join(where_clauses)

                    # Query to fetch details based on the identifier (assuming it's n.name)
                    # Match node by identifier, apply label filter, return details
                    cypher = (
                        f"MATCH (n) {where_clause} "
                        # Return n.name AS search_identifier to link back, and the required fields
                        f"RETURN n.name AS search_identifier, n.original_id AS original_id, n.name AS name, labels(n) AS node_labels "
                        f"LIMIT {len(candidate_identifiers)}" # Limit to max possible candidates
                    )
                    logger.debug(f"Executing Node Detail Cypher: {cypher} with params {params}")
                    result = session.run(cypher, params)

                    # Process results and map back using the search_identifier (which is n.name)
                    temp_detailed_candidates = {}
                    for record in result:
                        search_identifier = record['search_identifier']
                        if search_identifier in vector_candidates: # Check against original vector results (which used n.name)
                            # Check if original_id was fetched successfully
                            if record['original_id'] is None:
                                logger.warning(f"Node with name '{search_identifier}' lacks an original_id property. Skipping.")
                                continue

                            temp_detailed_candidates[search_identifier] = {
                                "name": record['name'], # Actual name property
                                "original_id": record['original_id'], # Actual original_id property
                                "labels": record['node_labels'],
                                "score": vector_candidates[search_identifier], # Get score from vector search results
                                "source": "vector"
                            }
                        else:
                            # This case should be less likely now, but kept for robustness
                            logger.warning(f"Identifier '{search_identifier}' found in DB detail query but not in initial vector candidates map. This might indicate an issue.")

                    # Reconstruct detailed_candidates preserving original vector order/scores
                    detailed_candidates = {}
                    for identifier in candidate_identifiers: # Iterate in original vector score order (using names)
                        if identifier in temp_detailed_candidates:
                            detailed_candidates[identifier] = temp_detailed_candidates[identifier]

            except (ServiceUnavailable, AuthError, CypherSyntaxError, Exception) as e:
                logger.error(f"Neo4j error during node detail fetch for vector candidates: {e}", exc_info=True) # Added exc_info
                # Proceed without details if DB fetch fails?
                # For now, return empty if details cannot be fetched.
                detailed_candidates = {}
            finally:
                if driver:
                    driver.close()
        else:
             logger.info("No candidates found from vector search meeting criteria.")

        # 3. Format Output (already sorted by vector score implicitly)
        # Convert dict to list, already sorted by score due to vector search result order
        # and threshold logic, plus reordering after DB fetch
        sorted_candidates = list(detailed_candidates.values())[:k] # Take top K valid candidates

        final_result = {
            "best_match": sorted_candidates[0] if sorted_candidates else None,
            "candidates": sorted_candidates
        }
        logger.info(f"Final resolution result: {final_result}")

        # 4. Persist result to LMDB cache (if available)
        if getattr(self, "_lmdb_env", None):
            try:
                with self._lmdb_env.begin(write=True) as txn:
                    txn.put(cache_key, json.dumps(final_result).encode())
            except Exception as e:
                logger.warning(f"Failed to write NameResolverTool result to LMDB cache: {e}")
        return final_result

    def _run(self, free_name: str, target_labels: List[str] | None = None, top_k: int = 5) -> Dict[str, Any]:
        """Execute the name resolution."""
        if not free_name:
            return {"error": "Input 'free_name' cannot be empty."}

        # Convert list to tuple for caching (or keep as list if json handles it)
        labels_tuple = tuple(sorted(target_labels)) if target_labels else None

        try:
            return self._perform_search(free_name, labels_tuple, top_k)
        except Exception as e:
            logger.error(f"Failed to resolve name '{free_name}': {e}", exc_info=True)
            return {"error": f"An unexpected error occurred during name resolution: {e}"}

    async def _arun(self, free_name: str, target_labels: List[str] | None = None, top_k: int = 5) -> Dict[str, Any]:
        """Asynchronous execution (placeholder)."""
        # Requires async implementation for Neo4j and potentially vector search
        logger.warning("_arun (async name resolver) is not implemented. Falling back to sync.")
        # Convert list to tuple for caching
        labels_tuple = tuple(sorted(target_labels)) if target_labels else None
        return self._run(free_name, labels_tuple, top_k)

    # Make the tool instance hashable so it can be used as part of the
    # cache key when functools.lru_cache is applied to instance methods.
    # Using the object's identity is sufficient here because each tool
    # instance maintains its own internal state and should not be grouped
    # with any other instance for caching purposes.
    def __hash__(self) -> int:  # noqa: D401, WPS615
        """Return an identity‑based hash so the instance is hashable."""
        return id(self)

# Example Usage (Requires Neo4j with APOC and optional vector setup)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import os

    # Get Neo4j connection details with defaults for testing
    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    cache_dir = os.environ.get("NR_CACHE_DIR", ".cache/name_resolver") # Example cache dir

    # Optional: Specify labels for index build/filtering if needed
    index_labels = ['Catalyst', 'Chemical', 'BasicMaterial'] # Example labels

    if not password:
        logger.info("Error: NEO4J_PASSWORD environment variable must be set.")
    elif not NEO4J_AVAILABLE:
        logger.info("Error: neo4j package not installed.")
    else:
        logger.info(f"Testing NameResolverTool connection to {uri}...")

        if not VECTOR_SEARCH_AVAILABLE:
            logger.info("Warning: faiss-cpu or sentence-transformers not installed. Vector search will be disabled.")

        try:
            # Initialize without full-text parameters
            tool = NameResolverTool(
                neo4j_uri=uri, neo4j_user=user, neo4j_password=password,
                node_labels_for_index=index_labels, # Pass labels if you want vector index specific to them
                cache_dir=cache_dir
                # cosine_similarity_threshold=0.6 # Optionally override threshold
            )

            test_name = "Catalyst A" # Example name
            logger.info(f"\nResolving name: '{test_name}' (Target: Catalyst)")
            result = tool.run({"free_name": test_name, "target_labels": ["Catalyst"], "top_k": 3})
            logger.info("\nResult:")
            logger.info(json.dumps(result, indent=2))

            test_typo = "MFI" # Example typo/alias
            logger.info(f"\nResolving name: '{test_typo}' (No target label)")
            result_typo = tool.run({"free_name": test_typo, "top_k": 3})
            logger.info("\nResult (typo):")
            logger.info(json.dumps(result_typo, indent=2))

            test_material = "silica" # Example basic material
            logger.info(f"\nResolving name: '{test_material}' (Target: BasicMaterial)")
            result_material = tool.run({"free_name": test_material, "target_labels": ["BasicMaterial"], "top_k": 3})
            logger.info("\nResult (material):")
            logger.info(json.dumps(result_material, indent=2))

        except (ImportError, ConnectionError, RuntimeError, ValueError) as e:
            logger.info(f"\nTest failed: {e}")
        except Exception as e:
            logger.info(f"\nTest failed with unexpected error: {e}")
            import traceback
            traceback.logger.info_exc() 