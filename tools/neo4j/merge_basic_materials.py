import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import faiss
import networkx as nx
import lmdb
from neo4j import GraphDatabase, Driver, Session, Transaction, Result
from tqdm import tqdm
import time
from collections import defaultdict
import json

# Attempt to import project-specific modules
try:
    from CATDA.models.models import get_model, set_debug_mode
    from CATDA.models.embedding import EntityEmbedding
except ImportError:
    logging.warning("Could not import from CATDA. Assuming script is run standalone.")
    # Add dummy implementations or raise a more specific error if needed
    # For now, let the script fail later if these are truly required and not found.
    pass 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_LLM_MODEL = 'google_gemini-2.5-flash-preview-04-17' # Changed default as requested in prompt description but not args
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_SIMILARITY_THRESHOLD = 0.75
LLM_CACHE_TRANSLATION_PREFIX = b"translation_" # Bytes prefix for translation cache keys
LLM_CACHE_SAME_SUBSTANCE_PREFIX = b"same_" # Bytes prefix for same substance cache keys
LLM_CACHE_CANONICAL_PREFIX = b"canonical_" # Bytes prefix for canonical name cache keys

# --- Neo4j Functions ---

def get_neo4j_driver(uri: str, user: str, password: Optional[str]) -> Optional[Driver]:
    """Establishes a connection to the Neo4j database."""
    if not password:
        logger.error("Neo4j password is required. Set NEO4J_PASSWORD environment variable or provide --neo4j_password.")
        return None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logger.info(f"Successfully connected to Neo4j at {uri}")
        return driver
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        return None

def fetch_basic_materials(tx: Transaction) -> List[Tuple[int, str]]:
    """Fetches node ID and name for all BasicMaterial nodes."""
    query = "MATCH (m:BasicMaterial) RETURN id(m) AS id, m.name AS name"
    result = tx.run(query)
    # Ensure name is treated as string, default to empty string if null/missing
    data = [(record["id"], str(record["name"] or '')) for record in result]
    fetched_count = len(data)
    # Filter out nodes with empty names AFTER fetching, log count
    data = [(id, name) for id, name in data if name and name.strip()]
    filtered_count = len(data)
    logger.info(f"Fetched {fetched_count} BasicMaterial nodes. Kept {filtered_count} nodes with non-empty names.")
    return data

def merge_nodes_in_neo4j(tx: Transaction, node_ids: List[int], canonical_name: str):
    """Merges specified nodes in Neo4j using APOC and sets the canonical name."""
    if not node_ids or len(node_ids) < 2:
        logger.warning(f"Skipping merge for canonical name '{canonical_name}' as it involves less than 2 nodes.")
        return

    merge_query = """
    MATCH (n) WHERE id(n) IN $ids
    WITH collect(n) AS ns
    CALL apoc.refactor.mergeNodes(ns, {
        properties: {name: 'overwrite', `.*`: 'discard'}, 
        mergeRels: true
    }) YIELD node
    SET node.name = $canonical_name // Ensure canonical name is explicitly set
    SET node:BasicMaterial       // Ensure label is retained/set
    RETURN id(node) as merged_node_id, node.name as final_name
    """
    try:
        result = tx.run(merge_query, ids=node_ids, canonical_name=canonical_name)
        summary = result.consume()
        logger.info(f"Merged {len(node_ids)} nodes into one with canonical name '{canonical_name}'. Counters: {summary.counters}")
    except Exception as e:
        logger.error(f"Failed to merge nodes {node_ids} with canonical name '{canonical_name}': {e}")
        # Re-raise or handle as needed, maybe indicate failure to the caller
        raise # Propagate the error for handling in the main loop

def update_or_merge_node_name(tx: Transaction, node_id: int, new_name: str, original_name: str):
    """
    Updates the name of a node to new_name.
    If another BasicMaterial node already exists with new_name, merges the current node into it.
    Otherwise, simply sets the name.
    Returns True if an operation (update or merge) was performed, False otherwise.
    """
    if new_name == original_name:
        logger.debug(f"Skipping update for Node ID {node_id}, English name '{new_name}' matches original.")
        return False # No operation needed

    query = """
    // Find the node to potentially update/merge
    MATCH (current:BasicMaterial) WHERE id(current) = $node_id
    WITH current
    // Look for an *existing* different node with the target name
    OPTIONAL MATCH (existing:BasicMaterial {name: $new_name})
    WHERE id(existing) <> $node_id
    WITH current, existing LIMIT 1 // Ensures we only find at most one target to merge into

    // Execute conditionally based on whether 'existing' was found
    CALL apoc.do.case(
        [
            // Condition 1: An existing node with the target name IS found
            existing IS NOT NULL,
                // Action 1: Merge 'current' into 'existing'
                'WITH $existing as target, $current as to_merge
                 CALL apoc.refactor.mergeNodes([target, to_merge], {
                     properties: { name: "overwrite", `.*`: "discard" },
                     mergeRels: true
                 }) YIELD node
                 SET node:BasicMaterial // Ensure label
                 RETURN "merged" AS action, id(node) as final_id, node.name as final_name'
        ],
        // Default Action (Condition 1 is false -> existing IS NULL): Update name
        'WITH $current as node_to_update
         SET node_to_update.name = $new_name
         RETURN "updated" AS action, id(node_to_update) as final_id, node_to_update.name as final_name',
        // Parameters for the query
        {existing: existing, current: current, new_name: $new_name}
    ) YIELD value

    RETURN value.action as action_taken, value.final_id as final_node_id, value.final_name as final_node_name
    """
    try:
        result: Result = tx.run(query, node_id=node_id, new_name=new_name)
        record = result.single()
        if record:
            action = record["action_taken"]
            final_id = record["final_node_id"]
            final_name = record["final_node_name"]
            if action == "merged":
                logger.debug(f"Merged Node ID {node_id} ('{original_name}') into existing Node ID {final_id} with name '{final_name}'")
                return True # A merge happened
            elif action == "updated":
                logger.debug(f"Updated name for Node ID {node_id} from '{original_name}' to '{final_name}'")
                return True # An update happened
            else:
                 logger.warning(f"Unexpected action '{action}' for node ID {node_id} when setting name to '{new_name}'")
                 return False # Unexpected outcome
        else:
            # This might happen if the initial MATCH (current) fails, which shouldn't occur if node_id is valid
            logger.warning(f"No action taken for node ID {node_id} when attempting to set name to '{new_name}'. Node might not exist or query failed silently.")
            return False # No operation confirmed

    except Exception as e:
        logger.error(f"Failed operation for node ID {node_id} (original: '{original_name}') setting name to '{new_name}': {e}")
        # Let the transaction handle rollback on error
        raise # Propagate exception to roll back the transaction in main

# --- Embedding and FAISS Functions ---

def get_embeddings(embedder: EntityEmbedding, names: List[str]) -> np.ndarray:
    """Generates embeddings for a list of names."""
    embeddings = []
    if not names:
         logger.warning("No names provided for embedding generation.")
         return np.array([])
         
    logger.info(f"Generating embeddings for {len(names)} names using model '{embedder.model}'...")
    # Use a single call to get_embeddings if available and efficient
    if hasattr(embedder, 'get_embeddings'): # Check if a batch method exists
        try:
            embeddings = embedder.get_embeddings(names) # Assumes it returns a list of lists/arrays or handles errors internally
            # Filter out None results if the batch method returns them for errors
            valid_embeddings = [emb for emb in embeddings if emb is not None]
            if len(valid_embeddings) != len(names):
                 logger.warning(f"Embedding generation resulted in {len(valid_embeddings)} embeddings for {len(names)} names. Some failed.")
            embeddings = valid_embeddings
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}. Falling back to individual generation.")
            embeddings = [] # Reset and try individual below
    
    # Fallback to individual generation if batch method doesn't exist or failed
    if not embeddings:
        for name in tqdm(names, desc="Generating Embeddings Individually"):
            try:
                if not name or not name.strip():
                    logger.warning(f"Skipping empty or whitespace-only name during embedding: '{name}'")
                    continue

                emb = embedder.get_embedding(name)
                if emb:
                    embeddings.append(emb)
                else:
                    logger.warning(f"Could not generate embedding for name: '{name}'. Skipping.")

            except Exception as e:
                logger.error(f"Error getting embedding for name '{name}': {e}")

    if not embeddings:
        logger.error("No embeddings were generated. Cannot proceed.")
        return np.array([])

    embeddings_np = np.array(embeddings).astype('float32')
    # Check for inconsistent dimensions, although EntityEmbedding should handle this
    if len(embeddings_np.shape) != 2:
        logger.error(f"Embeddings have inconsistent shapes. Cannot create numpy array. Shape: {embeddings_np.shape}")
        # Attempt to diagnose: print shapes of first few embeddings
        # for i, emb in enumerate(embeddings[:5]): logger.debug(f"Emb {i} shape: {np.array(emb).shape}")
        return np.array([])

    # Normalize embeddings for cosine similarity with IndexFlatIP
    faiss.normalize_L2(embeddings_np)
    logger.info(f"Generated and normalized {embeddings_np.shape[0]} embeddings with dimension {embeddings_np.shape[1]}.")
    return embeddings_np

def find_similar_pairs(embeddings: np.ndarray, k: int = 10, threshold: float = 0.75) -> List[Tuple[int, int, float]]:
    """Finds similar pairs using FAISS."""
    if embeddings.shape[0] < 2:
        logger.warning("Not enough embeddings ({embeddings.shape[0]}) to build FAISS index or find pairs.")
        return []

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Using Inner Product (cosine similarity on normalized vectors)
    try:
        index.add(embeddings)
        logger.info(f"Built FAISS IndexFlatIP with {index.ntotal} vectors.")
    except Exception as e:
         logger.error(f"FAISS index.add failed: {e}")
         return [] # Cannot proceed without index

    logger.info(f"Searching for top {k} neighbors for each vector...")
    try:
        distances, indices = index.search(embeddings, k)
        logger.info("FAISS search completed.")
    except Exception as e:
         logger.error(f"FAISS index.search failed: {e}")
         return [] # Cannot proceed without search results


    pairs = set()
    num_vectors = embeddings.shape[0]
    for i in range(num_vectors):
        for j_idx, neighbor_idx in enumerate(indices[i]):
            # Skip self-comparison and invalid indices (-1), ensure neighbor_idx is within bounds
            if neighbor_idx == i or neighbor_idx < 0 or neighbor_idx >= num_vectors:
                continue

            similarity = distances[i][j_idx]

            # Filter by threshold
            if similarity >= threshold:
                # Add pair with smaller index first to avoid duplicates (i, j) and (j, i)
                pair = tuple(sorted((i, neighbor_idx)))
                # Store highest similarity found for a given pair
                existing_sim = next((sim for p_i, p_j, sim in pairs if (p_i, p_j) == pair), -1)
                if similarity > existing_sim:
                     if existing_sim != -1: # Remove existing pair if found
                         pairs.remove((pair[0], pair[1], existing_sim))
                     pairs.add((*pair, similarity))

    # Convert set of tuples to list
    candidate_pairs = list(pairs)
    logger.info(f"Found {len(candidate_pairs)} unique candidate pairs above similarity threshold {threshold}.")
    # Sort by similarity descending (optional, but can be useful)
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)
    return candidate_pairs


# --- LLM Interaction Functions ---

def llm_translate_to_english(name: str, llm, db_txn) -> str:
    """Translates a given name to English using LLM and caches the result."""
    if not name or not name.strip():
        logger.warning("Skipping translation for empty name.")
        return "" # Return empty string for empty input

    cache_key = LLM_CACHE_TRANSLATION_PREFIX + name.encode('utf-8')

    # Check cache first
    cached_value = db_txn.get(cache_key)
    if cached_value is not None:
        try:
            translated_name = cached_value.decode('utf-8')
            logger.debug(f"Cache hit for translation: '{name}' -> '{translated_name}'")
            return translated_name
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to decode cached translation for key {cache_key}: {e}. Re-querying LLM.")

    prompt = f"""Translate the following chemical or material name into its most common English equivalent.
            Respond ONLY with the English translation, nothing else. If it is already in english, output as it is.
            
            Name: {name}
            English Translation:"""

    try:
        if hasattr(llm, 'invoke'):
            response_content = llm.invoke(prompt).content.strip()
            # Basic cleaning: remove potential quotes
            response_content = response_content.strip('\'"')
            
            if not response_content: # Handle empty response from LLM
                 logger.warning(f"LLM provided empty translation for '{name}'. Returning original.")
                 response_content = name # Fallback to original name if translation fails

        else:
            logger.error("LLM object does not have an 'invoke' method. Cannot query for translation.")
            response_content = name # Fallback to original name

        # Cache the result (even if it's the fallback original name)
        try:
            value_to_cache = response_content.encode('utf-8')
            db_txn.put(cache_key, value_to_cache, overwrite=True)
            logger.debug(f"Cached translation: '{name}' -> '{response_content}'")
        except lmdb.TxnFullError:
            logger.error("LLM LMDB transaction full. Increase map_size. Failed to cache translation.")
        except Exception as e:
            logger.error(f"Failed to write translation to LLM LMDB cache for key {cache_key}: {e}")

        return response_content

    except Exception as e:
        logger.error(f"LLM invocation failed during translation for '{name}': {e}")
        return name # Fallback to original name on error

def llm_same_substance(name_a: str, name_b: str, llm, db_txn, retries: int = 1) -> bool:
    """Uses LLM to determine if two ENGLISH names refer to the same chemical substance, using LMDB cache."""
    # Assumes name_a and name_b are already translated English names
    if not name_a or not name_b: return False
    if name_a == name_b: return True

    # Use canonical cache key (sorted tuple of English names, stringified)
    cache_key = LLM_CACHE_SAME_SUBSTANCE_PREFIX + str(tuple(sorted((name_a, name_b)))).encode('utf-8')

    # Check cache first
    cached_value = db_txn.get(cache_key)
    if cached_value is not None:
        try:
            return json.loads(cached_value.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to decode cached 'same substance' value for key {cache_key}: {e}. Re-querying LLM.")

    # Prompt assumes English names are provided
    prompt = f"""You are an expert in chemical nomenclature and synthesis practice.

                Task: Decide whether **Name A** and **Name B** refer to the *same* chemical substance.
                These names should already be in English.

                Reply **YES** - and nothing else - when they are the same substance, even if
                  - one name is a hydrate, solution, or common synonym, or even has difference concentration/composition of the other
                    (e.g. "NaOH" vs "sodium hydroxide solution vs "1 mol/L NaOH solution" vs "2 mol/L NaOH solution").
                Reply **NO** - and nothing else - when they are different substances.

                Return exactly "YES" or "NO" in uppercase without punctuation, explanation, or extra words.

                Name A: {name_a}
                Name B: {name_b}
                """

    results = []
    for _ in range(retries):
        try:
            if hasattr(llm, 'invoke'):
                response_content = llm.invoke(prompt).content.strip().upper()
            else:
                logger.error("LLM object does not have an 'invoke' method. Cannot query for same substance check.")
                return False

            if "YES" in response_content:
                results.append(True)
            else:
                results.append(False)
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"LLM invocation failed for 'same substance' check '{name_a}' vs '{name_b}': {e}")
            return False

    if not results: return False
    final_answer = max(set(results), key=results.count)

    # Store result in LMDB cache
    try:
        value_to_cache = json.dumps(final_answer).encode('utf-8')
        db_txn.put(cache_key, value_to_cache, overwrite=True)
        logger.debug(f"Cached LLM same substance result for ({name_a}, {name_b}): {final_answer}")
    except lmdb.TxnFullError:
         logger.error("LLM LMDB transaction full. Increase map_size. Failed to cache LLM same substance result.")
    except Exception as e:
         logger.error(f"Failed to write to LLM LMDB cache for 'same substance' key {cache_key}: {e}")

    return final_answer


def llm_choose_canonical(aliases: List[str], llm, db_txn) -> str:
    """Uses LLM to choose the canonical ENGLISH name from a list of ENGLISH aliases, using LMDB cache."""
    # Assumes aliases are already translated English names
    if not aliases:
        logger.warning("Cannot choose canonical name from an empty list.")
        return ""
    # Ensure aliases are unique and sorted for consistent caching
    unique_sorted_aliases = tuple(sorted(list(set(name for name in aliases if name and name.strip()))))
    if not unique_sorted_aliases:
         logger.warning("Cannot choose canonical name from empty/invalid aliases.")
         return ""
    if len(unique_sorted_aliases) == 1:
        return unique_sorted_aliases[0]

    # Use canonical cache key (stringified tuple of English names)
    cache_key = LLM_CACHE_CANONICAL_PREFIX + str(unique_sorted_aliases).encode('utf-8')

    # Check cache first
    cached_value = db_txn.get(cache_key)
    if cached_value is not None:
        try:
            return cached_value.decode('utf-8')
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to decode cached canonical name for key {cache_key}: {e}. Re-querying LLM.")

    alias_str = "\n".join(f"- {alias}" for alias in unique_sorted_aliases)
    prompt = f"""The following English names all refer to the **same** chemical substance:
                {alias_str}

                What is the most common or standard English name for this substance?
                - If there is a common English name, prefer that.
                - Otherwise, provide the IUPAC name or another widely accepted standard English name.
                - Respond *only* with the chosen canonical name, nothing else.
                """
    try:
        if hasattr(llm, 'invoke'):
            response = llm.invoke(prompt).content.strip()
        else:
             logger.error("LLM object does not have an 'invoke' method. Cannot query for canonical name.")
             return unique_sorted_aliases[0] # Fallback

        response = response.strip('\'"*- ')

        if not response: # Handle empty response from LLM
            logger.warning(f"LLM provided empty canonical name for {unique_sorted_aliases}. Falling back to first alias.")
            response = unique_sorted_aliases[0]

        # Optional: Check if the response is one of the inputs
        if response not in unique_sorted_aliases:
            logger.warning(f"LLM chose canonical name '{response}' which was not in the original unique alias list {list(unique_sorted_aliases)}. Using it anyway.")

        # Store result in LMDB cache
        try:
            value_to_cache = response.encode('utf-8')
            db_txn.put(cache_key, value_to_cache, overwrite=True)
            logger.debug(f"Cached LLM canonical name for {unique_sorted_aliases}: {response}")
        except lmdb.TxnFullError:
             logger.error("LLM LMDB transaction full. Increase map_size. Failed to cache LLM canonical name result.")
        except Exception as e:
             logger.error(f"Failed to write to LLM LMDB cache for canonical name key {cache_key}: {e}")

        return response

    except Exception as e:
        logger.error(f"LLM invocation failed for choosing canonical name from {list(unique_sorted_aliases)}: {e}")
        return unique_sorted_aliases[0] # Fallback


# --- Clustering Function ---

def build_clusters(validated_pairs: List[Tuple[int, int]], num_nodes: int) -> List[Set[int]]:
    """Builds clusters of node indices using NetworkX based on validated pairs."""
    if not validated_pairs and num_nodes > 0:
        # If no pairs validated, each node is its own cluster
        return [{i} for i in range(num_nodes)]
    elif num_nodes == 0:
        return [] # No nodes, no clusters

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes)) # Add all nodes by index
    # Only add edges for validated pairs
    for i, j in validated_pairs:
         # Ensure indices are within the valid range before adding edge
         if 0 <= i < num_nodes and 0 <= j < num_nodes:
              G.add_edge(i, j)
         else:
              logger.warning(f"Invalid index in validated pair ({i}, {j}) for num_nodes={num_nodes}. Skipping edge.")


    clusters = list(nx.connected_components(G))
    logger.info(f"Grouped {num_nodes} items into {len(clusters)} clusters based on validated pairs.")
    return clusters


# --- Main Workflow ---

def main():
    parser = argparse.ArgumentParser(description="Translate, cluster, and merge BasicMaterial nodes in Neo4j based on name similarity.")
    parser.add_argument("--neo4j_uri", type=str, default=os.environ.get("NEO4J_URI", "neo4j://localhost:7687"), help="Neo4j URI")
    parser.add_argument("--neo4j_user", type=str, default=os.environ.get("NEO4J_USER", "neo4j"), help="Neo4j username")
    parser.add_argument("--neo4j_password", type=str, default=os.environ.get("NEO4J_PASSWORD"), help="Neo4j password (reads NEO4J_PASSWORD env var by default)")
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL, help=f"LLM model name (default: {DEFAULT_LLM_MODEL})")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL, help=f"Embedding model name (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--similarity_threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help=f"FAISS similarity threshold (default: {DEFAULT_SIMILARITY_THRESHOLD})")
    parser.add_argument("--faiss_k", type=int, default=10, help="Number of neighbors to retrieve in FAISS search (default: 10)")
    parser.add_argument("--llm_retries", type=int, default=1, help="Number of times to ask LLM for self-consistency check (default: 1)")
    parser.add_argument("--embedding_cache_db", type=str, default="./embedding_cache.lmdb", help="Path to the LMDB file for embedding caching (used by EntityEmbedding).")
    parser.add_argument("--llm_cache_db", type=str, default="./llm_cache.lmdb", help="Path to the LMDB file for LLM translation/validation/canonical caching.")
    parser.add_argument("--lmdb_map_size", type=int, default=1024**3, help="LMDB map size in bytes for LLM Cache (default: 1GB)")
    parser.add_argument("--dry_run", action='store_true', help="Perform all steps except Neo4j merge/update operations.")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging for models.")


    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        try:
            set_debug_mode(debug=True)
        except NameError:
            logger.warning("set_debug_mode function not found, cannot configure model debug logging.")
        logger.debug("Debug mode enabled.")

    # --- Initialization ---
    logger.info("--- Starting BasicMaterial Merge Workflow ---")
    driver = None
    llm_cache_env = None
    embedder = None
    llm = None

    try:
        # Initialize Neo4j Driver
        driver = get_neo4j_driver(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        if not driver: return

        # Initialize Embedding Model
        try:
            emb_cache_path = Path(args.embedding_cache_db)
            emb_cache_path.parent.mkdir(parents=True, exist_ok=True)
            embedder = EntityEmbedding(model=args.embedding_model, db_path=emb_cache_path)
            logger.info(f"Initialized embedding model: {args.embedding_model} (cache path: {emb_cache_path})")
        except NameError:
             logger.error("EntityEmbedding class not found. Exiting.")
             return
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return

        # Initialize LLM Cache (LMDB)
        try:
            llm_cache_path = Path(args.llm_cache_db)
            llm_cache_path.parent.mkdir(parents=True, exist_ok=True)
            llm_cache_env = lmdb.open(str(llm_cache_path), map_size=args.lmdb_map_size, writemap=True)
            logger.info(f"Initialized LMDB cache for LLM at: {args.llm_cache_db} with map size {args.lmdb_map_size / (1024**2):.0f} MB")
        except Exception as e:
            logger.error(f"Failed to initialize LMDB cache for LLM: {e}")
            logger.warning("Proceeding without LLM caching.")
            llm_cache_env = None # Ensure it's None if initialization failed

        # Initialize LLM Model
        try:
            llm = get_model(args.llm_model)
            logger.info(f"Initialized LLM model: {args.llm_model}")
        except NameError:
            logger.error("get_model function not found. Exiting.")
            return
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            return

        # --- Start Core Logic ---
        node_data: List[Tuple[int, str]] = []
        embeddings_np: np.ndarray = np.array([])
        candidate_pairs: List[Tuple[int, int, float]] = []
        validated_pairs: List[Tuple[int, int]] = []
        translations: Dict[int, str] = {} # Map Neo4j ID -> English Name
        english_names: List[str] = [] # List of English names, index-aligned with node_data
        merged_node_ids: Set[int] = set() # Keep track of nodes *consumed* by merges (cluster merges or single node merges)
        processed_node_ids: Set[int] = set() # Keep track of node ids processed in final update/merge step
        
        # --- Step 1: Fetch Data from Neo4j ---
        try:
            with driver.session() as session:
                node_data = session.execute_read(fetch_basic_materials)
        except Exception as e:
            logger.error(f"Error fetching data from Neo4j: {e}")
            return

        if not node_data:
            logger.info("No BasicMaterial nodes with names found. Exiting.")
            return

        # --- Step 1.5: Translate Names to English (using LLM Cache) ---
        logger.info(f"Translating {len(node_data)} names to English...")
        if llm_cache_env:
            try:
                with llm_cache_env.begin(write=True) as txn:
                    for node_id, original_name in tqdm(node_data, desc="Translating Names"):
                        english_name = llm_translate_to_english(original_name, llm, db_txn=txn)
                        if english_name:
                             translations[node_id] = english_name
                             english_names.append(english_name)
                        else:
                             translations[node_id] = original_name # Store original as fallback
                             english_names.append(original_name)
                             logger.warning(f"Using original name '{original_name}' for Node ID {node_id} due to translation issue.")
                logger.info(f"Finished translation. Got {len(english_names)} names (including fallbacks).")
            except lmdb.Error as e:
                logger.error(f"LMDB error during translation transaction: {e}. Cannot proceed.")
                return
            except Exception as e:
                logger.error(f"Unexpected error during translation block: {e}")
                return
        else:
            logger.error("LLM Caching is disabled, required for translation. Exiting.")
            return

        if not english_names or len(english_names) != len(node_data):
             logger.error(f"Translation resulted in {len(english_names)} names, expected {len(node_data)}. Exiting.")
             return


        # --- Step 2: Get Embeddings (using English Names) ---
        embeddings_np = get_embeddings(embedder, english_names)
        if embeddings_np.size == 0:
             logger.error("Embedding generation failed. Exiting.")
             return

        if embeddings_np.shape[0] != len(english_names):
             logger.error(f"Embedding count ({embeddings_np.shape[0]}) mismatch with English name count ({len(english_names)}). Re-check embedding logic or data alignment. Exiting.")
             return


        # Map local index (0 to N-1) to Neo4j ID and English Name
        node_ids_map = {i: node_id for i, (node_id, _) in enumerate(node_data)}
        node_name_map = {i: name for i, name in enumerate(english_names)} # Use ENGLISH names

        # --- Step 3: Vector Similarity Search (FAISS) ---
        candidate_pairs = find_similar_pairs(embeddings_np, k=args.faiss_k, threshold=args.similarity_threshold)
        
        clusters_to_merge_idx: List[Set[int]] = [] # Initialize here for broader scope

        if not candidate_pairs:
            logger.info("No candidate pairs found above similarity threshold using English names.")
        else:
             # --- Step 4: LLM Validation (using English Names) ---
             logger.info(f"Validating {len(candidate_pairs)} candidate pairs using LLM {args.llm_model}...")
             if llm_cache_env:
                 try:
                     with llm_cache_env.begin(write=True) as txn:
                         for idx_a, idx_b, similarity in tqdm(candidate_pairs, desc="LLM Validation"):
                             name_a = node_name_map.get(idx_a)
                             name_b = node_name_map.get(idx_b)

                             if not name_a or not name_b:
                                  logger.warning(f"Skipping validation for pair ({idx_a}, {idx_b}) due to missing English names.")
                                  continue

                             is_same = llm_same_substance(name_a, name_b, llm, db_txn=txn, retries=args.llm_retries)
                             if is_same:
                                 validated_pairs.append((idx_a, idx_b))
                                 logger.debug(f"Validated (EN): '{name_a}' == '{name_b}' (Sim: {similarity:.4f})")
                             else:
                                 logger.debug(f"Rejected (EN): '{name_a}' != '{name_b}' (Sim: {similarity:.4f})")
                     logger.info(f"LLM validated {len(validated_pairs)} pairs using English names and cache.")

                 except lmdb.Error as e:
                     logger.error(f"LMDB error during LLM validation transaction: {e}. Validation incomplete.")
                     return
                 except Exception as e:
                     logger.error(f"Unexpected error during LLM validation block: {e}")
                     return
             else:
                 logger.error("LLM Caching is disabled, required for validation. Exiting.")
                 return

             if not validated_pairs:
                 logger.info("No pairs validated by LLM. No cluster merging needed.")
             else:
                 # --- Step 5: Clustering ---
                 num_nodes_for_clustering = len(english_names)
                 clusters_by_index = build_clusters(validated_pairs, num_nodes_for_clustering)
                 clusters_to_merge_idx = [cluster for cluster in clusters_by_index if len(cluster) > 1]
                 logger.info(f"Found {len(clusters_to_merge_idx)} clusters with >1 node requiring merging.")

                 if not clusters_to_merge_idx:
                     logger.info("No clusters require merging after validation.")
                 else:
                     # --- Step 6 & 7: Choose Canonical English Name and Merge Clusters in Neo4j ---
                     if args.dry_run:
                         logger.warning("--- DRY RUN ENABLED: Skipping Neo4j cluster merge operations. ---")

                     # Track nodes involved in these cluster merges specifically
                     cluster_merged_node_ids: Set[int] = set() 
                     successful_cluster_merges = 0
                     failed_cluster_merges = 0

                     if llm_cache_env:
                         try:
                             with llm_cache_env.begin(write=True) as llm_txn:
                                for cluster_indices in tqdm(clusters_to_merge_idx, desc="Choosing Canonical Names & Merging Clusters"):
                                     cluster_neo4j_ids = [node_ids_map[idx] for idx in cluster_indices]
                                     cluster_english_names = [node_name_map[idx] for idx in cluster_indices]

                                     unique_english_names = sorted(list(set(name for name in cluster_english_names if name and name.strip())))
                                     if not unique_english_names:
                                         logger.warning(f"Skipping cluster merge for Neo4j IDs {cluster_neo4j_ids}; invalid names.")
                                         failed_cluster_merges += 1
                                         continue

                                     canonical_name = llm_choose_canonical(unique_english_names, llm, db_txn=llm_txn)
                                     if not canonical_name:
                                          logger.error(f"LLM failed canonical name selection for cluster (EN names: {unique_english_names}). Skipping merge.")
                                          failed_cluster_merges += 1
                                          continue

                                     logger.info(f"Cluster (Neo4j IDs: {cluster_neo4j_ids}): Aliases (EN) {unique_english_names} -> Canonical (EN): '{canonical_name}'")

                                     # Add all nodes in this cluster to the set *before* attempting merge
                                     cluster_merged_node_ids.update(cluster_neo4j_ids)

                                     if not args.dry_run:
                                         try:
                                             with driver.session() as session:
                                                 # merge_nodes_in_neo4j now raises Exception on failure
                                                 session.execute_write(merge_nodes_in_neo4j, cluster_neo4j_ids, canonical_name)
                                             successful_cluster_merges += 1
                                         except Exception:
                                             # Error logged within merge_nodes_in_neo4j
                                             logger.error(f"Cluster merge failed for {cluster_neo4j_ids}. Transaction rolled back.")
                                             failed_cluster_merges += 1
                                     else:
                                          successful_cluster_merges += 1 # Count as success for dry run stats
                         except lmdb.Error as e:
                             logger.error(f"LMDB error during canonical name/merging transaction: {e}. Merging stopped.")
                             return
                         except Exception as e:
                             logger.error(f"Unexpected error during cluster merging block: {e}")
                             return
                     else:
                         logger.error("LLM Caching is disabled, required for canonical name selection. Exiting.")
                         return

                     logger.info(f"Cluster merge summary: {successful_cluster_merges} successful merges attempted/performed, {failed_cluster_merges} failed/skipped.")
                     # Add successfully cluster-merged IDs to the main merged_node_ids set
                     # Note: We add ALL attempted IDs earlier to avoid processing them in Step 8
                     merged_node_ids.update(cluster_merged_node_ids)

        # --- Step 8: Update/Merge Names of Non-Clustered Nodes to English ---
        if not args.dry_run:
             logger.info("Updating/merging names of non-clustered nodes to their English translations...")
             single_node_updates = 0
             single_node_merges = 0
             single_node_skipped = 0 # Includes nodes already processed in clusters or names matching

             try:
                 with driver.session() as session:
                     # Use one transaction for all these updates/merges
                     with session.begin_transaction() as tx:
                         # Iterate through original node data
                         for node_id, original_name in tqdm(node_data, desc="Updating/Merging Non-Clustered Nodes"):
                              # Skip nodes already handled by cluster merges
                              if node_id in merged_node_ids:
                                   single_node_skipped += 1
                                   continue
                              # Skip nodes already processed in this step (e.g., merged into another single node)
                              if node_id in processed_node_ids:
                                   single_node_skipped +=1
                                   continue

                              english_name = translations.get(node_id)
                              
                              if not english_name:
                                   logger.warning(f"Skipping Node ID {node_id}, missing English translation (should have fallback).")
                                   single_node_skipped += 1
                                   processed_node_ids.add(node_id) # Mark as processed
                                   continue
                                   
                              # Call the function that handles update or merge
                              # Pass original_name for better logging
                              try:
                                  operation_performed = update_or_merge_node_name(tx, node_id, english_name, original_name)
                                  
                                  # Need to know if it was an update or merge for stats
                                  # The function logs details, let's trust its return for counting
                                  if operation_performed:
                                       # How to know if it was merge vs update from return? The function logs it.
                                       # For simplicity, let's just count total operations here. Refine if needed.
                                       single_node_updates += 1 # Treat both update/merge as an "update" action count
                                       
                                       # If a merge happened, the original node_id was consumed. Add to merged_node_ids
                                       # How to know? We need more info from update_or_merge_node_name or check DB state after.
                                       # Easier: The function logs merge. We can assume operation_performed means success for now.
                                       
                                  else: # Operation was skipped (e.g., name same as original)
                                       single_node_skipped += 1
                                  
                                  processed_node_ids.add(node_id) # Mark current node ID as processed

                              except Exception:
                                   # Error logged in update_or_merge_node_name, transaction will roll back.
                                   logger.error(f"Transaction will be rolled back due to error processing Node ID {node_id}.")
                                   # Re-raise to ensure rollback
                                   raise

                         tx.commit() # Commit transaction if loop completes without error
                 logger.info(f"Finished update/merge for non-clustered nodes: {single_node_updates} operations performed, {single_node_skipped} skipped.")
             except Exception as e:
                 logger.error(f"Transaction failed during update/merge of non-clustered nodes: {e}")
                 # Transaction automatically rolls back on exception
        else:
             logger.warning("--- DRY RUN ENABLED: Skipping update/merge of non-clustered node names. ---")
             update_candidates = 0
             potential_single_merges = 0 # Harder to estimate without running checks
             for node_id, original_name in node_data:
                  if node_id not in merged_node_ids: # Ignore nodes part of cluster merges
                       english_name = translations.get(node_id)
                       if english_name and english_name != original_name:
                            # We can count potential updates, but potential merges require checking Neo4j state
                            update_candidates += 1
             logger.info(f"{update_candidates} non-clustered nodes *might* have their names updated or be merged (dry run).")


        # --- Final Summary ---
        logger.info("--- BasicMaterial Merge Workflow Completed ---")
        initial_node_count = len(node_data)
        logger.info(f"Initial node count: {initial_node_count}")
        logger.info(f"Names translated to English (incl. fallbacks): {len(english_names)}")
        logger.info(f"Embeddings generated: {embeddings_np.shape[0] if embeddings_np.size > 0 else 0}")
        logger.info(f"Candidate pairs found by FAISS (threshold > {args.similarity_threshold}): {len(candidate_pairs)}")
        logger.info(f"Pairs validated by LLM: {len(validated_pairs)}")
        num_clusters_merged = len(clusters_to_merge_idx) # Number of clusters identified for merging
        logger.info(f"Clusters identified for merging (>1 node): {num_clusters_merged}")

        if 'successful_cluster_merges' in locals(): # Check if cluster merge block was executed
            if not args.dry_run:
                nodes_in_cluster_merges = len(cluster_merged_node_ids) # Nodes involved in attempted cluster merges
                cluster_merge_reduction = nodes_in_cluster_merges - successful_cluster_merges if successful_cluster_merges > 0 else 0
                
                logger.info(f"Cluster Merging: {successful_cluster_merges} ops performed, {failed_cluster_merges} failed/skipped. Reduction: {cluster_merge_reduction} nodes.")
                logger.info(f"Single Node Update/Merge: {single_node_updates if 'single_node_updates' in locals() else 'N/A'} operations performed, {single_node_skipped if 'single_node_skipped' in locals() else 'N/A'} skipped.")
                # Final node count is hard to calculate precisely without querying DB, due to potential single-node merges
                # logger.info(f"Estimated final node count: {initial_node_count - cluster_merge_reduction - potential_single_merges_count}")
            else:
                 nodes_in_cluster_merges = len(cluster_merged_node_ids) if 'cluster_merged_node_ids' in locals() else 0
                 potential_cluster_merge_reduction = nodes_in_cluster_merges - successful_cluster_merges if successful_cluster_merges > 0 else 0
                 logger.info(f"Cluster Merging (Dry Run): {successful_cluster_merges} ops skipped. Potential reduction: {potential_cluster_merge_reduction} nodes.")
                 logger.info(f"Single Node Update/Merge (Dry Run): {update_candidates if 'update_candidates' in locals() else 'N/A'} ops skipped.")

    except Exception as e:
         # Catch any unexpected errors in the main try block
         logger.exception(f"An unexpected error occurred during the workflow: {e}") # Use logger.exception to include traceback

    finally:
        # --- Cleanup ---
        logger.info("--- Starting Cleanup --- ")
        if driver:
             try:
                 driver.close()
                 logger.info("Closed Neo4j connection.")
             except Exception as e:
                 logger.error(f"Error closing Neo4j driver: {e}")
        if llm_cache_env:
            try:
                llm_cache_env.close()
                logger.info("Closed LLM cache LMDB environment.")
            except Exception as e:
                logger.error(f"Error closing LLM cache LMDB environment: {e}")
        logger.info("--- Cleanup Finished --- ")

if __name__ == "__main__":
    main()
