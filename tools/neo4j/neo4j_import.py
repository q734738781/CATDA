import json5 as json
import argparse
import logging
import os
from pathlib import Path
from neo4j import GraphDatabase, basic_auth

# --- Constants for Flattening Prefixes ---
# Define explicit prefixes for flattening common dictionary attributes
# This helps standardize query patterns (e.g., property_ vs properties_)
FLATTEN_PREFIX_MAP = {
    "properties": "property",           # Chemical/Catalyst properties
    "composition": "composition",      # Chemical/Catalyst composition (remains same)
    "conditions": "condition",         # Synthesis conditions
    "conditions_json": "condition",    # Testing conditions
    "results_json": "result",          # Testing results
    "results": "result"                # Testing results
    # Add other keys here if they contain dicts to be flattened with specific prefixes
}

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants for Node/Relationship Types ---
NODE_TYPE_MAP = {
    "chemical": "Chemical",
    "synthesis": "Synthesis",
    "testing": "Testing",
    "characterization": "Characterization",
    # New Types
    "catalyst": "Catalyst",  # Replaces Chemical for tested catalysts
    "basic_material": "BasicMaterial",
    # Corrected Types for Test-Specific Nodes - REMOVED
    # "test_condition": "TestCondition", # Removed
    # "test_performance_metric": "TestPerformanceMetric", # Removed
    "paper": "Paper",
    # "formula": "Formula" # Removed Formula type
}

# Define colors and styling for each node type in Neo4j Browser
NODE_STYLE_MAP = {
    "Chemical": {
        "color": "#66CCFF",  # Light blue
        "size": 80,
        "caption": "name"
    },
    "Catalyst": { # Style for chemicals identified as catalysts
        "color": "#FF6347",  # Tomato Red
        "border_color": "#CC0000",
        "size": 90,
        "caption": "name"
    },
    "Synthesis": {
        "color": "#FF9966",  # Orange
        "size": 65,
        "caption": "name"
    },
    "Testing": {
        "color": "#99CC66",  # Light green
        "size": 65,
        "caption": "id"
    },
    "Characterization": {
        "color": "#9370DB",  # Medium Purple
        "size": 65,
        "caption": "method_name"
    },
    # New Styles
    "BasicMaterial": {
        "color": "#DAA520", # Goldenrod
        "size": 50,
        "caption": "name"
    },
    # Corrected Styles for Test-Specific Nodes - REMOVED
    # "TestCondition": {
    #     "color": "#4682B4", # Steel Blue
    #     "size": 45,
    #     "caption": "original_id" # No single 'name' property
    # },
    # "TestPerformanceMetric": {
    #     "color": "#3CB371", # Medium Sea Green
    #     "size": 45,
    #     "caption": "original_id" # No single 'name' property
    # },
    "Paper": {
        "color": "#808080", # Gray
        "size": 100,
        "caption": "name"
    },
    # "Formula": { # Removed Formula style
    #     "color": "#BA55D3", # Medium Orchid
    #     "size": 40,
    #     "caption": "name"
    # },
     "UnknownNode": { # Default for safety
        "color": "#CCCCCC",
        "size": 50,
        "caption": "original_id"
    }
}


EDGE_TYPE_MAP = {
    "synthesis_input": "SYNTHESIS_INPUT",
    "synthesis_output": "SYNTHESIS_OUTPUT",
    "tested_in": "TESTED_IN",
    "characterized_in": "CHARACTERIZED_IN",
    # New Types
    "appear_in": "APPEAR_IN",
    "with_basicmaterial": "WITH_BASICMATERIAL", # Renamed from HAS_COMPONENT
    # Corrected Relationship targets/meaning - REMOVED Test specific relationships
    # "result_in": "RESULT_IN", # Testing -> TestPerformanceMetric - Removed
    # "tested_under": "TESTED_UNDER", # Testing -> TestCondition - Removed
    "related_to": "RELATED_TO" # Default fallback
    # "has_formula": "HAS_FORMULA" # Removed HAS_FORMULA relationship
}

def flatten_and_prepare_attributes(data: dict) -> dict:
    """
    Flattens dictionary attributes one level deep for Neo4j compatibility,
    creating keys like prefix_innerkey_unit using FLATTEN_PREFIX_MAP.
    Handles lists by JSON serialization.
    """
    prepared_props = {}
    # Keys handled elsewhere (identity, specific logic) or structural - skip entirely
    keys_to_skip_entirely = {
        "id", "type", "original_id",
        "is_catalyst", "is_tested_catalyst" # Flags used for logic, not stored directly this way
        # Removed: "composition", "chemical formula", "formula"
    }

    for key, value in data.items():
        if key in keys_to_skip_entirely:
            continue

        # --- Try flattening if value is a dictionary --- 
        if isinstance(value, dict):
            # Determine prefix using the map, fallback to original key
            prefix = FLATTEN_PREFIX_MAP.get(key)
            if prefix is None:
                # Fallback: Use original key name if not in the map
                # Remove _json suffix if present for consistency in fallbacks
                prefix = key
                if key.lower().endswith("_json"):
                    prefix = key[:-5]
                logger.debug(f"Key '{key}' not found in FLATTEN_PREFIX_MAP, using derived prefix '{prefix}_' for flattening.")
            
            prefix += "_" # Add underscore separator

            for inner_key, inner_value in value.items():
                final_value = None
                unit_suffix = "NoUnit" # Default suffix

                if isinstance(inner_value, dict) and "value" in inner_value:
                    # Expected structure: {"value": ..., "unit": ...} or {"value": ...}
                    final_value = inner_value.get("value")
                    unit = inner_value.get("unit")
                    if unit and isinstance(unit, str) and unit.strip():
                        # Basic sanitization: replace spaces and common problematic chars for Neo4j props
                        safe_unit = unit.strip().replace(' ', '_').replace('/', '-per-').replace('%', 'pct')
                        # Avoid overly long keys if unit is very long, maybe truncate or hash? For now, use sanitized.
                        unit_suffix = safe_unit
                elif isinstance(inner_value, (dict, list)):
                     # Fallback: Stringify complex types (nested dicts/lists)
                     try:
                         final_value = json.dumps(inner_value, ensure_ascii=False)
                     except TypeError:
                         final_value = str(inner_value) # Ultimate fallback
                else:
                    # Fallback: Use string representation for simple non-dict values within the dict
                    final_value = str(inner_value)

                # Basic sanitization for inner_key
                safe_inner_key = inner_key.strip().replace(' ', '_').replace('.', '_').replace('-', '_')
                flattened_key = f"{prefix}{safe_inner_key}_{unit_suffix}"

                # Add the flattened property
                prepared_props[flattened_key] = final_value

        elif isinstance(value, list):
            # Keep serializing lists
            try:
                prepared_props[key] = json.dumps(value, ensure_ascii=False)
            except TypeError as e:
                logger.warning(f"Could not serialize list attribute '{key}' to JSON: {e}. Storing as string representation.")
                prepared_props[key] = str(value)
        else:
            # Keep simple scalar values directly
            prepared_props[key] = value

    return prepared_props

def make_neo4j_safe(props: dict) -> dict:
    """
    Ensures a property dictionary only contains Neo4j-supported types:
    primitives (str, int, float, bool) or arrays of primitives. Any dicts or
    lists containing non-primitives are serialized to JSON strings.
    Also lightly sanitizes keys to avoid problematic characters.
    """
    safe_props = {}
    for key, value in props.items():
        safe_key = key.strip().replace(' ', '_').replace('.', '_').replace('-', '_')
        if isinstance(value, dict):
            try:
                safe_props[safe_key] = json.dumps(value, ensure_ascii=False)
            except Exception:
                safe_props[safe_key] = str(value)
        elif isinstance(value, list):
            # If the list contains any dicts/lists, serialize the whole list
            if any(isinstance(x, (dict, list)) for x in value):
                try:
                    safe_props[safe_key] = json.dumps(value, ensure_ascii=False)
                except Exception:
                    safe_props[safe_key] = str(value)
            else:
                safe_props[safe_key] = value
        else:
            safe_props[safe_key] = value
    # Raise a warning if the properties are not safe
    if not safe_props:
        logger.warning(f"Properties are not safe: {props}. If this happens frequently, check the data source for non-safe values (dict, list, etc.).")
    return safe_props

def import_graph_to_neo4j(driver, json_data: dict, paper_name: str, clear_db: bool):
    """Imports the graph structure from JSON data into Neo4j, creating additional nodes and relationships."""

    # Extract nodes and edges from all relevant sections
    synthesis_data = json_data.get("synthesis", {})
    testing_data = json_data.get("testing", {})
    characterization_data = json_data.get("characterization", {})
    catgraph_tree_data = json_data.get("catgraph_tree", {}) # Handle combined format

    nodes_data = (
        synthesis_data.get("nodes", [])
        + testing_data.get("nodes", [])
        + characterization_data.get("nodes", [])
        + catgraph_tree_data.get("nodes", [])
    )
    edges_data = (
        synthesis_data.get("edges", [])
        + testing_data.get("edges", [])
        + characterization_data.get("edges", [])
        + catgraph_tree_data.get("edges", [])
    )

    # --- Pre-process to find starting materials and catalysts ---
    all_node_ids = set()
    output_node_ids = set()
    catalyst_node_ids = set()

    for node in nodes_data:
        if isinstance(node, dict) and "id" in node:
            node_id = node["id"]
            all_node_ids.add(node_id)
            node_type = node.get("type", "").lower()
            raw_properties = node # Access properties directly for flag check
            # Handle potential is_tested_catalyst -> is_catalyst rename before checking
            if "is_tested_catalyst" in raw_properties:
                raw_properties["is_catalyst"] = raw_properties.pop("is_tested_catalyst")
            
            if (node_type == "chemical" and raw_properties.get("is_catalyst")) or node_type == "catalyst":
                catalyst_node_ids.add(node_id)


    for edge in edges_data:
        if isinstance(edge, dict) and edge.get("type") == "synthesis_output" and "target_id" in edge:
            output_node_ids.add(edge["target_id"])

    starting_material_ids = all_node_ids - output_node_ids
    eligible_for_bm_extraction_ids = starting_material_ids.union(catalyst_node_ids)
    logger.info(f"Identified {len(eligible_for_bm_extraction_ids)} nodes as potential starting materials or catalysts for BasicMaterial extraction.")
    # --- End Pre-processing ---


    with driver.session() as session:
        if clear_db:
            logger.info("Clearing existing database...")
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared.")

        # Create constraints and indexes first
        try:
            # Remove the old generic constraint if it exists
            # session.run("DROP CONSTRAINT node_original_id IF EXISTS")
            # Use modern syntax ON...ASSERT...UNIQUE
            session.run("CREATE CONSTRAINT chemical_original_id IF NOT EXISTS FOR (c:Chemical) REQUIRE c.original_id IS UNIQUE")
            session.run("CREATE CONSTRAINT catalyst_original_id IF NOT EXISTS FOR (cat:Catalyst) REQUIRE cat.original_id IS UNIQUE")
            session.run("CREATE CONSTRAINT synthesis_original_id IF NOT EXISTS FOR (s:Synthesis) REQUIRE s.original_id IS UNIQUE")
            session.run("CREATE CONSTRAINT testing_original_id IF NOT EXISTS FOR (t:Testing) REQUIRE t.original_id IS UNIQUE")
            session.run("CREATE CONSTRAINT characterization_original_id IF NOT EXISTS FOR (c:Characterization) REQUIRE c.original_id IS UNIQUE")
            session.run("CREATE CONSTRAINT unknown_original_id IF NOT EXISTS FOR (u:UnknownNode) REQUIRE u.original_id IS UNIQUE")
            session.run("CREATE CONSTRAINT paper_name IF NOT EXISTS FOR (p:Paper) REQUIRE p.name IS UNIQUE")
            session.run("CREATE CONSTRAINT material_name IF NOT EXISTS FOR (m:BasicMaterial) REQUIRE m.name IS UNIQUE")
            logger.info("Ensured constraints and indexes exist using modern syntax.")
        except Exception as e:
            logger.warning(f"Could not create/modify constraints/indexes (might exist, be dropped already, or insufficient permissions): {e}")

        # --- 1. Create Paper Node ---
        logger.info(f"Ensuring Paper node exists: {paper_name}")
        try:
            session.run("MERGE (p:Paper {name: $paper_name})", paper_name=paper_name)
        except Exception as e:
            logger.error(f"Failed to create/merge Paper node {paper_name}: {e}")
            raise RuntimeError(f"Could not ensure Paper node '{paper_name}' exists.") from e

        # --- 2. Create Nodes and New Related Nodes/Relationships ---
        logger.info(f"Processing {len(nodes_data)} nodes from source JSON...")
        created_node_count = 0
        skipped_node_count = 0

        for node in nodes_data:
            if not isinstance(node, dict) or "id" not in node or "type" not in node:
                logger.warning(f"Skipping invalid node data structure: {node}")
                skipped_node_count += 1
                continue

            json_node_id = node["id"] # Original ID from JSON
            node_type_str = node.get("type", "unknown").lower()
            is_catalyst_node = False # Flag to track if this becomes a Catalyst

            # Prepare properties, excluding standard/structural keys and basic material source keys
            raw_properties = {k: v for k, v in node.items() if k not in ["id", "type"]}

            # Handle potential is_tested_catalyst -> is_catalyst rename (already done in pre-processing, but check again for safety)
            if "is_tested_catalyst" in raw_properties:
                raw_properties["is_catalyst"] = raw_properties.pop("is_tested_catalyst")
            
            # Determine final Neo4j Label
            neo4j_label = NODE_TYPE_MAP.get(node_type_str)
            if not neo4j_label:
                logger.warning(f"Node {json_node_id} has unknown type '{node_type_str}'. Assigning 'UnknownNode'.")
                neo4j_label = "UnknownNode"
            
            # Default assumption: node is paper-specific
            is_paper_specific_node = True
            
            # Handle Catalyst case and check if it's a shared type
            if neo4j_label == "Chemical" and raw_properties.get("is_catalyst"):
                neo4j_label = "Catalyst" # Override label
                is_catalyst_node = True
                # Don't pop is_catalyst yet, needed for eligibility check below
            elif neo4j_label in ["BasicMaterial", "Paper"]:
                 is_paper_specific_node = False # These nodes are global
            
            # --- Determine if eligible for BasicMaterial extraction ---
            is_eligible_for_bm = json_node_id in eligible_for_bm_extraction_ids

            # --- Generate Neo4j ID and Prepare Final Properties ---
            if is_paper_specific_node:
                neo4j_original_id = f"{paper_name}_{json_node_id}"
                # Serialize AFTER potentially removing formula/composition
                # Pass raw_properties before potential modification for BasicMaterial extraction
                properties_for_node = flatten_and_prepare_attributes(raw_properties.copy())
                properties_for_node['original_id'] = neo4j_original_id # Composite ID for Neo4j uniqueness
                properties_for_node['json_id'] = json_node_id       # Original ID from JSON for reference
                properties_for_node['paper_name'] = paper_name       # Explicitly link to paper
            else:
                # For global nodes like BasicMaterial (defined directly in JSON)
                if neo4j_label == "BasicMaterial":
                    material_name = raw_properties.get("name", json_node_id)
                    if not material_name:
                        logger.warning(f"Skipping BasicMaterial node with missing name/id: {node}")
                        skipped_node_count += 1
                        continue
                    properties_for_node = {"name": material_name, "original_id": material_name}
                else:
                    logger.warning(f"Unhandled global node type '{neo4j_label}' for node: {node}")
                    skipped_node_count += 1
                    continue # Skip this node

            properties_for_node['original_type'] = node_type_str # Store original type for reference

            # Final defensive sanitation to ensure only Neo4j-supported property types
            properties_for_node = make_neo4j_safe(properties_for_node)

            # --- Create the main node ---
            if is_paper_specific_node:
                 node_create_query = f"MERGE (n:{neo4j_label} {{original_id: $id}}) SET n = $props RETURN n"
                 query_params = {"id": neo4j_original_id, "props": properties_for_node}
            elif neo4j_label == "BasicMaterial":
                 node_create_query = f"MERGE (m:BasicMaterial {{name: $name}}) SET m += $props RETURN m"
                 query_params = {"name": material_name, "props": properties_for_node}
            # Paper node already handled

            try:
                # Only run merge for paper-specific or BasicMaterial nodes here
                if is_paper_specific_node or neo4j_label == "BasicMaterial":
                    result = session.run(node_create_query, **query_params)
                    created_node_count += 1
            except Exception as e:
                 logger.error(f"Failed to create/merge node (Label: {neo4j_label}, JSON ID: {json_node_id}): {e}")
                 logger.debug(f"Query: {node_create_query}")
                 logger.debug(f"Params: {query_params}")
                 skipped_node_count += 1
                 continue # Skip related entity creation if base node failed

            # --- Create related nodes and relationships (only if main node succeeded) ---
            try:
                # Catalyst -> Paper (Relationship from Catalyst node)
                if is_catalyst_node and is_paper_specific_node: # Ensure it's the paper-specific catalyst node
                    rel_query = """
                    MATCH (c:Catalyst {original_id: $catalyst_neo4j_id})
                    MATCH (p:Paper {name: $paper_name})
                    MERGE (c)-[r:APPEAR_IN]->(p)
                    """
                    session.run(rel_query, catalyst_neo4j_id=neo4j_original_id, paper_name=paper_name)

                # --- Extract BasicMaterial (New Prioritized Logic) ---
                # Only for eligible starting materials/catalysts that are paper-specific
                if is_eligible_for_bm and is_paper_specific_node and neo4j_label in ["Chemical", "Catalyst"]:
                    basic_material_extracted = False

                    # 1. Check for Chemical Formula
                    formula_key = "chemical formula" if "chemical formula" in raw_properties else "formula"
                    if formula_key in raw_properties:
                        formula_value = raw_properties.get(formula_key)
                        if isinstance(formula_value, str) and formula_value.strip():
                            # Clean the formula string conservatively:
                            # - If '(' appears after the first character, drop trailing parenthetical notes
                            # - Otherwise, keep as-is but remove only fully wrapping parentheses
                            candidate = formula_value.strip()
                            if '(' in candidate and candidate.find('(') > 0:
                                candidate = candidate.split('(')[0].strip()
                            while len(candidate) >= 2 and candidate.startswith('(') and candidate.endswith(')'):
                                candidate = candidate[1:-1].strip()

                            cleaned_material_name = candidate
                            if cleaned_material_name:
                                material_name = cleaned_material_name
                                logger.info(f"Node {neo4j_original_id}: Using formula '{formula_value}' -> BasicMaterial '{material_name}'.")
                                bm_props = {"name": material_name, "original_id": material_name}
                                bm_merge_query = "MERGE (m:BasicMaterial {name: $name}) SET m += $props RETURN m"
                                try:
                                    session.run(bm_merge_query, name=material_name, props=bm_props)
                                except Exception as bm_e:
                                    logger.error(f"Failed to MERGE BasicMaterial '{material_name}' from formula for '{neo4j_original_id}': {bm_e}")
                                else:
                                    # Create WITH_BASICMATERIAL Relationship
                                    rel_props = {'source_property': 'formula'}
                                    rel_props = make_neo4j_safe(rel_props)
                                    rel_query = """
                                    MATCH (chem {original_id: $chem_neo4j_id})
                                    MATCH (mat:BasicMaterial {name: $mat_name})
                                    MERGE (chem)-[r:WITH_BASICMATERIAL]->(mat)
                                    SET r = $props
                                    """
                                    try:
                                        session.run(rel_query, chem_neo4j_id=neo4j_original_id, mat_name=material_name, props=rel_props)
                                        basic_material_extracted = True
                                    except Exception as rel_e:
                                        logger.error(f"Failed to create WITH_BASICMATERIAL rel (formula) '{neo4j_original_id}' -> '{material_name}': {rel_e}")
                            else:
                                logger.warning(f"Node {neo4j_original_id}: Empty material name after cleaning formula '{formula_value}'.")
                        else:
                            logger.debug(f"Node {neo4j_original_id}: Invalid 'formula' property: {formula_value}.")

                    # 2. Check for Composition (if no formula processed)
                    if not basic_material_extracted and "composition" in raw_properties:
                        composition = raw_properties.get("composition")
                        if isinstance(composition, dict) and composition:
                            logger.info(f"Node {neo4j_original_id}: Processing composition for BasicMaterials.")
                            for raw_material_name, comp_details in composition.items():
                                cleaned_material_name = raw_material_name.split('(')[0].strip()
                                if not cleaned_material_name:
                                    logger.warning(f"Empty material name after cleaning composition key '{raw_material_name}' for node '{neo4j_original_id}'. Skipping.")
                                    continue

                                material_name = cleaned_material_name
                                bm_props = {"name": material_name, "original_id": material_name}
                                bm_merge_query = "MERGE (m:BasicMaterial {name: $name}) SET m += $props RETURN m"
                                try:
                                    session.run(bm_merge_query, name=material_name, props=bm_props)
                                except Exception as bm_e:
                                    logger.error(f"Failed to MERGE BasicMaterial '{material_name}' from composition key '{raw_material_name}': {bm_e}")
                                    continue # Skip relationship if material node fails

                                # Create WITH_BASICMATERIAL Relationship
                                rel_props = {'source_property': 'composition'}
                                if isinstance(comp_details, dict):
                                    rel_props['value'] = comp_details.get('value')
                                    rel_props['unit'] = comp_details.get('unit')
                                else:
                                    rel_props['value'] = str(comp_details) # Fallback

                                rel_props = make_neo4j_safe(rel_props)

                                rel_query = """
                                MATCH (chem {original_id: $chem_neo4j_id})
                                MATCH (mat:BasicMaterial {name: $mat_name})
                                MERGE (chem)-[r:WITH_BASICMATERIAL]->(mat)
                                SET r = $props
                                """
                                try:
                                    session.run(rel_query, chem_neo4j_id=neo4j_original_id, mat_name=material_name, props=rel_props)
                                    basic_material_extracted = True # Mark as extracted even if only one component worked
                                except Exception as rel_e:
                                    logger.error(f"Failed to create WITH_BASICMATERIAL rel (composition) '{neo4j_original_id}' -> '{material_name}': {rel_e}")
                        elif composition: # Composition exists but is not a dict
                             logger.warning(f"Node {neo4j_original_id}: 'composition' property is not a dictionary: {composition}. Cannot extract BasicMaterials.")


                    # 3. Check for Name (if no formula or composition processed, AND node is NOT a Catalyst)
                    if not basic_material_extracted and "name" in raw_properties and neo4j_label != "Catalyst":
                        node_name = raw_properties.get("name")
                        if isinstance(node_name, str) and node_name.strip():
                            cleaned_material_name = node_name.split('(')[0].strip()
                            if cleaned_material_name:
                                material_name = cleaned_material_name
                                logger.info(f"Node {neo4j_original_id}: Using name '{node_name}' -> BasicMaterial '{material_name}'.")
                                bm_props = {"name": material_name, "original_id": material_name}
                                bm_merge_query = "MERGE (m:BasicMaterial {name: $name}) SET m += $props RETURN m"
                                try:
                                    session.run(bm_merge_query, name=material_name, props=bm_props)
                                except Exception as bm_e:
                                    logger.error(f"Failed to MERGE BasicMaterial '{material_name}' from name for '{neo4j_original_id}': {bm_e}")
                                else:
                                    # Create WITH_BASICMATERIAL Relationship
                                    rel_props = {'source_property': 'name'}
                                    rel_props = make_neo4j_safe(rel_props)
                                    rel_query = """
                                    MATCH (chem {original_id: $chem_neo4j_id})
                                    MATCH (mat:BasicMaterial {name: $mat_name})
                                    MERGE (chem)-[r:WITH_BASICMATERIAL]->(mat)
                                    SET r = $props
                                    """
                                    try:
                                        session.run(rel_query, chem_neo4j_id=neo4j_original_id, mat_name=material_name, props=rel_props)
                                    except Exception as rel_e:
                                        logger.error(f"Failed to create WITH_BASICMATERIAL rel (name) '{neo4j_original_id}' -> '{material_name}': {rel_e}")
                            else:
                                logger.warning(f"Node {neo4j_original_id}: Empty material name after cleaning name '{node_name}'.")
                        else:
                             logger.warning(f"Node {neo4j_original_id}: Invalid 'name' property: {node_name}.")


            except Exception as e:
                # Log errors happening during related entity/relationship creation
                logger.error(f"Failed creating related entities/rels for node (Label: {neo4j_label}, JSON ID: {json_node_id}): {e}")
                # Continue processing other nodes

        logger.info(f"Finished processing nodes from source. Created/Merged: {created_node_count}, Skipped: {skipped_node_count}")

        # --- 3. Create Original Edges (Relationships) defined in JSON ---
        logger.info(f"Processing {len(edges_data)} relationships from source JSON...")
        created_edge_count = 0
        skipped_edge_count = 0
        for edge in edges_data:
            if not isinstance(edge, dict) or not all(k in edge for k in ["type", "source_id", "target_id"]):
                logger.warning(f"Skipping invalid edge data structure: {edge}")
                skipped_edge_count += 1
                continue

            edge_type_str = edge.get("type", "related_to").lower()
            # Skip creating BasicMaterial relationships here as they are handled during node processing
            if edge_type_str == "with_basicmaterial" or edge_type_str == "has_component": # Check old name too for safety
                continue

            neo4j_rel_type = EDGE_TYPE_MAP.get(edge_type_str)
            if not neo4j_rel_type:
                logger.warning(f"Edge type '{edge_type_str}' not found in EDGE_TYPE_MAP. Using default 'RELATED_TO'. Edge: {edge}")
                neo4j_rel_type = EDGE_TYPE_MAP["related_to"]

            source_json_id = edge["source_id"]
            target_json_id = edge["target_id"]

            # Generate composite IDs for source and target based on current paper context
            source_neo4j_id = f"{paper_name}_{source_json_id}"
            target_neo4j_id = f"{paper_name}_{target_json_id}"

            # Prepare properties for the relationship
            raw_properties = {k: v for k, v in edge.items() if k not in ["id", "type", "source_id", "target_id"]}
            properties = flatten_and_prepare_attributes(raw_properties)
            properties['original_id'] = edge.get("id", f"{source_neo4j_id}_{target_neo4j_id}_{edge_type_str}") # Make original rel ID context aware too
            properties['original_type'] = edge_type_str
            properties['paper_name'] = paper_name

            # Final defensive sanitation to ensure only Neo4j-supported property types
            properties = make_neo4j_safe(properties)

            # Create Cypher query for relationship creation using composite IDs
            # Match nodes based on their original_id which includes the paper name prefix
            query = (
                f"MATCH (source {{original_id: $source_id}}), (target {{original_id: $target_id}}) "
                f"MERGE (source)-[r:{neo4j_rel_type}]->(target) "
                f"SET r = $props"
            )
            try:
                result = session.run(query, source_id=source_neo4j_id, target_id=target_neo4j_id, props=properties)
                summary = result.consume()
                # Use consume() summary for more accurate counts if needed
                created_edge_count += summary.counters.relationships_created + summary.counters.properties_set
            except Exception as e:
                # Log which relationship failed
                logger.error(f"Failed to create relationship {source_json_id} -[{neo4j_rel_type}]-> {target_json_id} (Paper: {paper_name}): {e}")
                logger.debug(f"Query: MATCH (source {{original_id: '{source_neo4j_id}'}}), (target {{original_id: '{target_neo4j_id}'}}) MERGE (source)-[r:{neo4j_rel_type}]->(target) SET r = ...")
                logger.debug(f"Edge properties: {properties}")
                skipped_edge_count += 1

        logger.info(f"Finished processing relationships from source. Created/Merged approx: {created_edge_count}, Skipped: {skipped_edge_count}")


def main():
    parser = argparse.ArgumentParser(description="Import CatGraphNX JSON data into Neo4j, creating enhanced graph structure.")
    parser.add_argument("input_path", type=str, help="Path to the input JSON/JSON5 file or a directory containing such files.")
    parser.add_argument("--neo4j_uri", type=str, default=os.environ.get("NEO4J_URI", "neo4j://localhost:7687"), help="Neo4j URI (default: neo4j://localhost:7687 or NEO4J_URI env var).")
    parser.add_argument("--neo4j_user", type=str, default=os.environ.get("NEO4J_USER", "neo4j"), help="Neo4j username (default: neo4j or NEO4J_USER env var).")
    parser.add_argument("--neo4j_password", type=str, default=os.environ.get("NEO4J_PASSWORD"), help="Neo4j password (reads NEO4J_PASSWORD env var by default).")
    parser.add_argument("--clear", action="store_true", help="Clear the Neo4j database before importing the first file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--paper_name", type=str, help="Specify a single name for the Paper node IF processing a single file (overrides auto-detection).")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    if not args.neo4j_password:
        logger.error("Neo4j password not provided. Set the NEO4J_PASSWORD environment variable or use the --neo4j_password argument.")
        return

    input_path = Path(args.input_path)
    files_to_process = []

    if input_path.is_file():
        if input_path.suffix.lower() in [".json", ".json5"]:
            files_to_process.append(input_path)
            logger.info(f"Processing single input file: {input_path}")
        else:
            logger.error(f"Input file is not a .json or .json5 file: {input_path}")
            return # Return here if single file is not json/json5
    elif input_path.is_dir():
        logger.info(f"Processing directory: {input_path}")
        json_files = list(input_path.glob('*.json'))
        json5_files = list(input_path.glob('*.json5'))
        files_to_process = sorted(json_files + json5_files) # Combine and sort
        if not files_to_process:
            logger.warning(f"No .json or .json5 files found in directory: {input_path}")
            return # Return if no files found in directory
        logger.info(f"Found {len(files_to_process)} JSON/JSON5 files to process.")
    else:
        logger.error(f"Input path is neither a valid file nor a directory: {input_path}")
        return

    db_cleared = False
    total_files_processed = 0
    total_files_failed = 0

    driver = None
    try:
        auth = basic_auth(args.neo4j_user, args.neo4j_password)
        driver = GraphDatabase.driver(args.neo4j_uri, auth=auth)
        try:
            driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {args.neo4j_uri}")
        except Exception as conn_e:
            logger.error(f"Failed initial connection verification to Neo4j: {conn_e}")
            logger.error("Please check Neo4j connection details, credentials, and ensure the server is running and accessible.")
            return

        with driver: # Use context manager for the driver
            for file_path in files_to_process:
                logger.info(f"--- Processing file: {file_path.name} ---")
                try:
                    # Determine paper name
                    paper_name_for_file = args.paper_name
                    if not paper_name_for_file or len(files_to_process) > 1:
                        paper_name_for_file = file_path.stem
                        try: # Attempt heuristic splitting
                           paper_name_for_file = paper_name_for_file.split('_')[0]
                        except IndexError:
                           pass # Keep the full stem if no underscore
                        logger.info(f"Using automatically determined paper name: {paper_name_for_file}")
                    else:
                        logger.info(f"Using specified paper name: {paper_name_for_file}")

                    # Read JSON data
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        logger.info(f"Successfully loaded JSON data from {file_path.name}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from {file_path.name}. Invalid JSON5 syntax? Error: {e}")
                        total_files_failed += 1
                        continue
                    except Exception as e:
                        logger.error(f"Error reading JSON file {file_path.name}: {e}")
                        total_files_failed += 1
                        continue

                    # Validate JSON structure (basic check)
                    if not isinstance(json_data, dict) or not any(k in json_data for k in ["synthesis", "testing", "catgraph_tree"]):
                        logger.error(f"Error: JSON data in {file_path.name} must be a dictionary with at least one of 'synthesis', 'testing', or 'catgraph_tree' keys.")
                        total_files_failed += 1
                        continue

                    # Handle DB clearing
                    should_clear = args.clear and not db_cleared

                    # Import data
                    import_graph_to_neo4j(driver, json_data, paper_name_for_file, should_clear)

                    if should_clear:
                        db_cleared = True

                    logger.info(f"Import process completed for {file_path.name}.")
                    total_files_processed += 1

                except Exception as import_e:
                    logger.error(f"Failed to import data from {file_path.name}: {import_e}")
                    logger.exception("Import error details:")
                    total_files_failed += 1
                    # Continue with the next file

            logger.info(f"--- Overall Summary ---")
            logger.info(f"Successfully processed: {total_files_processed} files")
            logger.info(f"Failed to process: {total_files_failed} files")

    except Exception as e:
        logger.error(f"A critical error occurred during the Neo4j operation: {e}")
        logger.exception("Process error details:")
    # Driver closed automatically by 'with' statement

if __name__ == "__main__":
    main()