import logging
import re
import json
from typing import Type, List, Dict, Any, Tuple
import os
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

# Vector Search Imports (Optional)
try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    logging.warning("faiss-cpu or sentence-transformers not found. Vector search in FieldNameResolverTool is disabled.")
    SentenceTransformer = None
    faiss = None
    np = None

# Neo4j Imports (needed for dynamic property key fetching)
try:
    from neo4j import GraphDatabase, basic_auth
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None
    ServiceUnavailable = None
    AuthError = None
    NEO4J_AVAILABLE = False
    logging.warning("neo4j package not found. FieldNameResolverTool may have limited functionality.")

logger = logging.getLogger(__name__)

# --- Tool Input Schema ---
class FieldNameResolverInput(BaseModel):
    """Input schema for the FieldNameResolverTool."""
    natural_language_phrase: str = Field(description="The natural language phrase representing a metric, condition, or property (e.g., 'conversion', 'reaction temperature').")
    top_k: int = Field(default=10, description="Maximum number of candidate property keys to return.")

class FieldNameResolverTool(BaseTool):
    """
    Translates a natural language phrase for a metric or condition into one or more
    canonical property keys used in the Neo4j graph, using dynamic data from the graph.
    Builds a vector index over property keys fetched from Neo4j on initialization.
    Uses vector search for primary matching, with optional regex fallback.
    """
    name: str = "FieldNameResolver"
    description: str = (
        "Translate a natural language phrase (e.g., 'reaction temperature', 'conversion') into canonical graph property keys. "
        "Uses vector search over known keys. Returns best match and candidates."
    )
    args_schema: Type[BaseModel] = FieldNameResolverInput

    # Vector search config (Placeholders - requires external setup)
    _vector_embedding_model: str = PrivateAttr(default='all-MiniLM-L6-v2') # Model used for indexing keys/synonyms
    _vector_model_instance: Any = PrivateAttr(default=None)
    _vector_index_instance: Any = PrivateAttr(default=None)
    _vector_id_map: List[str] | None = PrivateAttr(default=None) # List of property keys in index order
    _vector_index_path: str | None = PrivateAttr(default=None)
    _vector_id_map_path: str | None = PrivateAttr(default=None)
    _cosine_similarity_threshold: float = PrivateAttr(default=0.65)

    # User-provided regex mapping (optional)
    _regex_map_path: str | None = PrivateAttr(default=None)
    _user_regex_patterns: List[Tuple[re.Pattern, str]] | None = PrivateAttr(default=None)
    _regex_map_version: float | None = PrivateAttr(default=None)

    # Optional Neo4j connection (for future validation)
    _neo4j_uri: str | None = PrivateAttr(default=None)
    _neo4j_user: str | None = PrivateAttr(default=None)
    _neo4j_password: str | None = PrivateAttr(default=None)

    # Make the tool instance hashable so it can be used as part of the
    # cache key when functools.lru_cache is applied to instance methods.
    # Using the object's identity is sufficient here because each tool
    # instance maintains its own internal state and should not be grouped
    # with any other instance for caching purposes.
    def __hash__(self) -> int:  # noqa: D401, WPS615
        """Return an identity‑based hash so the instance is hashable."""
        return id(self)

    def __init__(
        self,
        # Neo4j details are now required
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        # Vector search options remain optional
        vector_embedding_model: str = 'all-MiniLM-L6-v2',
        cosine_similarity_threshold: float = 0.4,
        cache_dir: str | None = None,
        regex_map_path: str | None = None,
        **kwargs
    ):
        """Initialize with Neo4j connection details and vector search parameters."""
        super().__init__(**kwargs)

        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver is not available. Please install it using `pip install neo4j`.")
        if not neo4j_uri or not neo4j_user or not neo4j_password:
            raise ValueError("neo4j_uri, neo4j_user, and neo4j_password are required for FieldNameResolverTool.")

        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._vector_embedding_model = vector_embedding_model
        self._cosine_similarity_threshold = cosine_similarity_threshold
        self._regex_map_path = regex_map_path

        # Determine cache paths
        self._vector_index_path = None
        self._vector_id_map_path = None
        if cache_dir:
            vec_dir = os.path.join(cache_dir, "propkey_resolver_vectors")
            os.makedirs(vec_dir, exist_ok=True)
            self._vector_index_path = os.path.join(vec_dir, "pkr.faiss")
            self._vector_id_map_path = os.path.join(vec_dir, "pkr_id_map.json")
        else:
            logger.warning("Cache directory not specified. Vector index and LMDB cache will not be persisted.")

        # Load optional user regex mapping
        self._load_user_regex_mapping()

        # Load or build vector search index
        logger.info("Initializing FieldNameResolverTool vector search index... \n If this is the first time you are using this tool or graph has changed, it may take a while.")
        self._initialize_vector_search()
        logger.info("FieldNameResolverTool vector search index initialized.")

        # Persistent cache setup with LMDB
        try:
            import lmdb
            self._lmdb_env = None
            if cache_dir:
                cache_path = os.path.join(cache_dir, 'property_key_resolver')
                os.makedirs(cache_path, exist_ok=True)
                self._lmdb_env = lmdb.open(
                    cache_path,
                    map_size=1 << 30,  # 1 GiB
                    subdir=True,
                    max_dbs=1,
                    readonly=False,
                    lock=False,
                )
        except ImportError:
            self._lmdb_env = None
            logger.warning("lmdb not installed – FieldNameResolverTool will run without persistent caching.")

    def _load_user_regex_mapping(self) -> None:
        """Load and compile user-provided regex->canonical key mappings from JSON file (optional)."""
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
            # Support dict {pattern: canonical_key} or list of {pattern/name}
            if isinstance(data, dict):
                items = data.items()
            elif isinstance(data, list):
                items = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    pat = item.get("pattern") or item.get("regex")
                    key = item.get("key") or item.get("name") or item.get("canonical")
                    if pat and key:
                        items.append((pat, key))
            else:
                logger.warning("Unsupported JSON format for regex mapping. Expect dict or list of objects.")
                return

            for entry in items:
                if isinstance(entry, tuple):
                    pattern_str, canonical_key = entry
                else:
                    pattern_str, canonical_key = entry[0], entry[1]
                try:
                    compiled.append((re.compile(pattern_str, re.IGNORECASE), str(canonical_key)))
                except re.error as rex:
                    logger.warning(f"Invalid regex pattern skipped in FieldNameResolverTool: {pattern_str!r} ({rex})")
            self._user_regex_patterns = compiled
            try:
                self._regex_map_version = os.path.getmtime(self._regex_map_path)
            except Exception:
                self._regex_map_version = None
            logger.info(f"Loaded {len(self._user_regex_patterns)} user regex mappings for FieldNameResolverTool.")
        except Exception as e:
            logger.warning(f"Failed to load regex mapping file for FieldNameResolverTool: {e}")

    def _initialize_vector_search(self):
        """Load or build the vector index for property keys."""
        if not VECTOR_SEARCH_AVAILABLE or not NEO4J_AVAILABLE:
            logger.warning("Missing vector search or Neo4j dependencies. Property key vector search disabled.")
            return

        # --- 1. Fetch Current Property Keys from Neo4j ---
        current_keys: List[str] = []
        driver = None
        try:
            self._vector_model_instance = SentenceTransformer(self._vector_embedding_model)
            driver = GraphDatabase.driver(self._neo4j_uri, auth=basic_auth(self._neo4j_user, self._neo4j_password))
            with driver.session() as session:
                # Use apoc.meta.data() to fetch schema information
                schema_info = {"labels": {}, "relationships": set(), "properties": set()}
                meta_result = session.run("CALL apoc.meta.schema()")
                meta_data = meta_result.single()[0] # APOC returns a nested dict
                property_keys = set()

                for label, data in meta_data.items():
                    if data['type'] == 'node':
                        props = sorted(list(data.get('properties', {}).keys()))  # Sort properties here
                        schema_info["labels"][label] = props
                        schema_info["properties"].update(props)
                    elif data['type'] == 'relationship':
                        schema_info["relationships"].add(label)  # APOC uses rel type as key
                        # Optionally extract relationship properties if needed
                        rel_props = sorted(list(data.get('properties', {}).keys()))
                        schema_info["properties"].update(rel_props)  # Add rel props if desired

                current_keys = sorted(schema_info["properties"])

            logger.debug(f"Fetched {len(current_keys)} unique property keys from Neo4j using apoc.meta.data().")
        except Exception as e:
            logger.error(f"Failed to fetch property keys/load model for index check: {e}")
            self._vector_model_instance = None
            return
        finally:
            if driver:
                driver.close()

        if not current_keys:
            logger.warning("No property keys found in Neo4j. Skipping vector index build.")
            return

        # --- 2. Check Cached Keys ---
        cached_keys: List[str] | None = None
        cache_valid = False
        if self._vector_id_map_path and os.path.exists(self._vector_id_map_path) and self._vector_index_path and os.path.exists(self._vector_index_path):
            try:
                with open(self._vector_id_map_path, "r", encoding="utf-8") as f:
                    cached_keys = json.load(f)
                if current_keys == cached_keys:
                    cache_valid = True
                else:
                    logger.info("Neo4j property keys have changed. Rebuilding FieldNameResolverTool vector index.")
            except Exception as e:
                logger.warning(f"Could not read cached property key map ({self._vector_id_map_path}): {e}. Rebuilding index.")

        # --- 3. Load or Rebuild ---
        if cache_valid and cached_keys:
            try:
                logger.info(f"Loading cached FAISS index ({self._vector_index_path}) for FieldNameResolverTool.")
                self._vector_index_instance = faiss.read_index(self._vector_index_path)
                self._vector_id_map = cached_keys
                logger.info(f"Successfully loaded {len(self._vector_id_map)} property key vectors from cache.")
            except Exception as e:
                logger.error(f"Failed to load cached FAISS index ({self._vector_index_path}): {e}. Attempting rebuild.")
                cache_valid = False

        if not cache_valid:
            logger.info("Building FieldNameResolverTool vector index from Neo4j property keys...")
            if not self._vector_model_instance:
                logger.error("Embedding model not loaded. Cannot build property key vector index.")
                return
            try:
                embeddings = self._vector_model_instance.encode(current_keys, show_progress_bar=False, normalize_embeddings=True).astype("float32")
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)

                self._vector_index_instance = index
                self._vector_id_map = current_keys

                if self._vector_index_path and self._vector_id_map_path:
                    faiss.write_index(index, self._vector_index_path)
                    with open(self._vector_id_map_path, "w", encoding="utf-8") as f:
                        json.dump(current_keys, f, indent=2)
                    logger.info(f"Built and saved new property key vector index ({len(current_keys)} keys) to {self._vector_index_path}.")
                else:
                    logger.warning("Cache directory not configured. Property key vector index will not be persisted.")

            except Exception as e:
                logger.error(f"Failed during property key vector index build/save: {e}", exc_info=True)
                self._vector_index_instance = None
                self._vector_id_map = None

    # Persistent cache using LMDB (if available)
    def _resolve_key(self, phrase: str, k: int) -> Dict[str, Any]:
        cache_key_dict = {"phrase": phrase, "k": k, "regex_ver": self._regex_map_version}
        cache_key = json.dumps(cache_key_dict, sort_keys=True).encode()

        # Try reading from cache first
        if getattr(self, "_lmdb_env", None):
            with self._lmdb_env.begin(write=False) as txn:
                cached_bytes = txn.get(cache_key)
                if cached_bytes:
                    try:
                        return json.loads(cached_bytes.decode())
                    except Exception as e:
                        logger.warning(f"Corrupted LMDB entry for FieldNameResolverTool: {e}. Recomputing.")
        """Internal resolution logic using regex (high priority) and vector search, cached."""
        logger.info(f"Resolving property key for phrase: '{phrase}', k={k}")
        candidates = {}
        phrase_lower = phrase.lower()

        # 0. User-provided regex mapping (higher priority)
        if self._user_regex_patterns:
            matched_keys: List[str] = []
            for pattern, key in self._user_regex_patterns:
                try:
                    if pattern.search(phrase_lower):
                        matched_keys.append(key)
                except Exception:
                    continue
            if matched_keys:
                # Add regex matches as top candidates with highest score
                for key in dict.fromkeys(matched_keys):
                    if key not in candidates:
                        candidates[key] = {"key": key, "score": 1.0, "source": "regex"}
                sorted_candidates = sorted(candidates.values(), key=lambda x: x['score'], reverse=True)[:k]
                canonical = sorted_candidates[0]['key'] if sorted_candidates else None
                result = {"canonical": canonical, "candidates": sorted_candidates}
                if getattr(self, "_lmdb_env", None):
                    try:
                        with self._lmdb_env.begin(write=True) as txn:
                            txn.put(cache_key, json.dumps(result).encode())
                    except Exception as e:
                        logger.warning(f"Failed to write FieldNameResolverTool result to LMDB cache: {e}")
                return result

        # 1. Vector Search (if available)
        if self._vector_model_instance and self._vector_index_instance and self._vector_id_map:
            try:
                query_embedding = self._vector_model_instance.encode([phrase_lower])[0]
                # FAISS with cosine similarity needs index built with inner product (IndexFlatIP)
                # and normalized embeddings. Search returns inner product scores.
                # If using L2 index, distances need conversion.
                # Assuming IndexFlatIP and normalized embeddings for cosine similarity search:
                scores, indices = self._vector_index_instance.search(np.array([query_embedding]), k * 2)

                resolved_keys = []
                for i, idx in enumerate(indices[0]):
                    if idx < 0 or idx >= len(self._vector_id_map): continue
                    score = scores[0][i] # Cosine similarity score
                    if score >= self._cosine_similarity_threshold:
                        key = self._vector_id_map[idx] # Key is directly the string from the map
                        if key not in candidates or score > candidates[key]['score']:
                            resolved_keys.append(key)
                            candidates[key] = {
                                "key": key,
                                "score": float(score), # Ensure score is float
                                "source": "vector"
                            }
                logger.debug(f"Vector search resolved keys: {resolved_keys}")
            except Exception as e:
                logger.error(f"Error during property key vector search: {e}", exc_info=True)
        elif self._vector_index_path:
             logger.warning("Property key vector search specified but not initialized.")

        # 2. No built-in regex fallback anymore; rely solely on user-provided mappings above

        # 3. Rank and Format
        sorted_candidates = sorted(candidates.values(), key=lambda x: x['score'], reverse=True)
        top_k_candidates = sorted_candidates[:k]

        # Select a single "canonical" key (highest score)
        canonical = top_k_candidates[0]['key'] if top_k_candidates else None

        result = {
            "canonical": canonical,
            "candidates": top_k_candidates
        }
        logger.info(f"Property key resolution result: {result}")

        # Persist result
        if getattr(self, "_lmdb_env", None):
            try:
                with self._lmdb_env.begin(write=True) as txn:
                    txn.put(cache_key, json.dumps(result).encode())
            except Exception as e:
                logger.warning(f"Failed to write FieldNameResolverTool result to LMDB cache: {e}")
        return result

    def _run(self, natural_language_phrase: str, top_k: int = 3) -> Dict[str, Any]:
        """Execute property key resolution."""
        if not natural_language_phrase:
            return {"error": "Input 'natural_language_phrase' cannot be empty."}

        try:
            # Use lowercased phrase for caching consistency
            return self._resolve_key(natural_language_phrase.lower(), top_k)
        except Exception as e:
            logger.error(f"Failed to resolve property key for '{natural_language_phrase}': {e}", exc_info=True)
            return {"error": f"An unexpected error occurred during property key resolution: {e}"}

    async def _arun(self, natural_language_phrase: str, top_k: int = 3) -> Dict[str, Any]:
        """Asynchronous execution (placeholder)."""
        logger.warning("_arun (async property key resolver) is not implemented. Falling back to sync.")
        return self._run(natural_language_phrase, top_k)

# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import os

    # Get Neo4j connection details from environment variables (with defaults for testing)
    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687") # Default URI for testing
    user = os.environ.get("NEO4J_USER", "neo4j") # Default user for testing
    password = os.environ.get("NEO4J_PASSWORD") # No default for password
    cache_dir = os.environ.get("PKR_CACHE_DIR", ".cache") # Example cache dir

    # Password is still required
    if not password:
        logger.info("Error: NEO4J_PASSWORD environment variable must be set.")
    elif not NEO4J_AVAILABLE:
        logger.info("Error: neo4j package not installed.")
    elif not VECTOR_SEARCH_AVAILABLE:
        logger.info("Error: faiss-cpu or sentence-transformers not installed. Vector search is not working.")
    else:
        # Vector search dependencies are available
        logger.info("Testing FieldNameResolverTool (Vector Search Enabled)...")
        try:
            tool = FieldNameResolverTool(
                neo4j_uri=uri,
                neo4j_user=user,
                neo4j_password=password,
                cache_dir=cache_dir
            )

            test_phrase_1 = "Ethylbenzene Conversion Rate"
            logger.info(f"\nResolving key for: '{test_phrase_1}'")
            result1 = tool.run({"natural_language_phrase": test_phrase_1, "top_k": 5})
            logger.info("Result 1:")
            logger.info(json.dumps(result1, indent=2))

            test_phrase_2 = "Yield of product A"
            logger.info(f"\nResolving key for: '{test_phrase_2}'")
            result2 = tool.run({"natural_language_phrase": test_phrase_2})
            logger.info("Result 2:")
            logger.info(json.dumps(result2, indent=2))

            test_phrase_3 = "Surface Area (BET)"
            logger.info(f"\nResolving key for: '{test_phrase_3}'")
            # This likely needs vector search or a specific regex
            result3 = tool.run({"natural_language_phrase": test_phrase_3})
            logger.info("Result 3 (Needs Vector/Better Regex):")
            logger.info(json.dumps(result3, indent=2))

        except ImportError as e:
             logger.info(f"\nTest failed due to missing dependency: {e}")
        except ConnectionError as e:
            logger.info(f"\nTest failed due to Neo4j connection error: {e}")
        except Exception as e:
            logger.info(f"\nTest failed with unexpected error: {e}")
            import traceback
            traceback.logger.info_exc() 