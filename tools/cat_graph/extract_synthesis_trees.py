import json
import argparse
import logging
import os
from pathlib import Path
from collections import defaultdict
import networkx as nx
import copy
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_full_synthesis_graph(synthesis_data: dict) -> tuple[nx.DiGraph, dict, dict]:
    """Builds a NetworkX DiGraph from the synthesis section, including all nodes and edges."""
    graph = nx.DiGraph()
    nodes_dict = {node['id']: node for node in synthesis_data.get("nodes", [])}
    edges_list = synthesis_data.get("edges", [])

    # Add all nodes from the synthesis section
    for node_id, node_data in nodes_dict.items():
        graph.add_node(node_id, **node_data)
        logger.debug(f"Added node: {node_id}")

    # Add all edges from the synthesis section
    for edge in edges_list:
        source = edge.get("source_id")
        target = edge.get("target_id")
        edge_id = edge.get("id", f"{source}_{target}") # Use provided id or generate one

        if source in graph and target in graph:
            # Create a copy of edge attributes and remove 'id' if present to avoid conflict
            edge_attrs = edge.copy()
            edge_attrs.pop('id', None) # Remove 'id' key safely

            graph.add_edge(source, target, id=edge_id, **edge_attrs)
            logger.debug(f"Added edge: {source} -> {target} (ID: {edge_id})")
        else:
            logger.warning(f"Skipping edge {edge_id} due to missing node: source='{source}', target='{target}'")

    return graph, nodes_dict, {edge.get("id", f"{edge.get('source_id')}_{edge.get('target_id')}"): edge for edge in edges_list}


def extract_catgraph_tree(graph: nx.DiGraph, target_catalyst_id: str, all_nodes_data: dict, all_edges_data: dict, all_testing_nodes_data: dict, all_testing_edges_data: dict, all_char_nodes_data: dict | None = None, all_char_edges_data: dict | None = None) -> dict | None:
    """Extracts the synthesis tree and appends relevant testing nodes/edges.
    
    Also adds synthesis levels (backward from catalyst) to synthesis nodes only.
    """
    if target_catalyst_id not in graph:
        logger.warning(f"Target catalyst ID '{target_catalyst_id}' not found in the synthesis graph.")
        return None

    # --- Step 1: Find all nodes and edges in the full backward history --- 
    tree_nodes_ids = set()
    tree_edges_ids = set()
    queue = [target_catalyst_id] # Use a list as a queue for BFS
    visited_nodes_traversal = set()

    logger.debug(f"Starting backward traversal from: {target_catalyst_id}")
    while queue:
        current_node_id = queue.pop(0)
        if current_node_id in visited_nodes_traversal:
            continue
        visited_nodes_traversal.add(current_node_id)
        # Ensure the node actually exists in the original data before adding
        if current_node_id in all_nodes_data:
            tree_nodes_ids.add(current_node_id)
            logger.debug(f"  Traversal: Added node {current_node_id}")

            # Find predecessors and add edges
            for predecessor_id in graph.predecessors(current_node_id):
                if predecessor_id not in visited_nodes_traversal:
                    queue.append(predecessor_id)
                
                # Check if edge exists and add its ID
                if graph.has_edge(predecessor_id, current_node_id):
                    edge_data = graph.get_edge_data(predecessor_id, current_node_id)
                    if edge_data and 'id' in edge_data:
                        edge_id = edge_data['id']
                        # Ensure the edge exists in original data before adding
                        if edge_id in all_edges_data:
                            tree_edges_ids.add(edge_id)
                            logger.debug(f"    Traversal: Added edge {predecessor_id} -> {current_node_id} (ID: {edge_id})")
                        else:
                             logger.warning(f"Traversal: Edge ID {edge_id} found in graph but not in all_edges_data.")
                    else:
                        logger.warning(f"Traversal: Could not get edge ID for {predecessor_id} -> {current_node_id}")
        else:
             logger.debug(f"  Traversal: Node {current_node_id} (reached via predecessor) not in all_nodes_data, skipping.")

    if not tree_nodes_ids:
        logger.warning(f"Could not find any valid nodes for catalyst '{target_catalyst_id}' during traversal.")
        return None
    logger.debug(f"Traversal complete. Found {len(tree_nodes_ids)} nodes and {len(tree_edges_ids)} edges.")

    # --- Step 2: Build the subgraph containing only the extracted nodes and edges --- 
    tree_graph = nx.DiGraph()
    tree_graph.add_nodes_from(tree_nodes_ids)
    for edge_id in tree_edges_ids:
        edge_data = all_edges_data[edge_id]
        u = edge_data.get("source_id")
        v = edge_data.get("target_id")
        if u in tree_graph and v in tree_graph: # Ensure both nodes are part of the tree
            tree_graph.add_edge(u, v)
        else:
             logger.warning(f"Skipping edge {edge_id} during tree_graph construction as node not found (u='{u}', v='{v}').")
    logger.debug(f"Built tree_graph with {tree_graph.number_of_nodes()} nodes and {tree_graph.number_of_edges()} edges.")

    # --- Step 3: Identify Synthesis Nodes in the Tree --- 
    synthesis_nodes_in_tree = {nid for nid in tree_nodes_ids if all_nodes_data[nid].get('type') == 'synthesis'}

    node_levels = {} # Store final levels: {synthesis_node_id: level}
    if not synthesis_nodes_in_tree:
        logger.warning(f"No synthesis nodes found in the history of catalyst '{target_catalyst_id}'. Cannot calculate levels.")
    elif target_catalyst_id not in tree_graph: 
         logger.error(f"Target catalyst '{target_catalyst_id}' is not in the constructed tree_graph. Cannot calculate levels.")
    else:
        # --- Step 4: Calculate Raw Distances (Backward BFS on Full Tree Graph) --- 
        raw_distances = {node_id: -1 for node_id in tree_nodes_ids} # distance from catalyst
        queue_bfs = [(target_catalyst_id, 0)] # Queue stores (node_id, distance)
        raw_distances[target_catalyst_id] = 0
        visited_bfs = {target_catalyst_id} # Keep track of visited nodes for BFS

        logger.debug(f"Starting backward BFS for raw distance calculation from: {target_catalyst_id} on tree_graph")
        head = 0
        while head < len(queue_bfs):
            current_node_id, current_dist = queue_bfs[head]
            head += 1
            logger.debug(f"  Raw Distance BFS: Visiting {current_node_id} at distance {current_dist}")

            # Iterate through predecessors in the tree_graph (includes chemical nodes)
            for predecessor_id in tree_graph.predecessors(current_node_id):
                if predecessor_id not in visited_bfs:
                    visited_bfs.add(predecessor_id)
                    raw_distances[predecessor_id] = current_dist + 1
                    queue_bfs.append((predecessor_id, current_dist + 1))
                    logger.debug(f"    Raw Distance BFS: Added predecessor {predecessor_id} at distance {current_dist + 1}")
        logger.debug(f"Raw Distance BFS complete.")

        # --- Step 5: Rank Synthesis Nodes by Distance --- 
        reached_synthesis_nodes = {nid for nid in synthesis_nodes_in_tree if raw_distances.get(nid, -1) != -1}
        
        if reached_synthesis_nodes:
            # Create list of (distance, node_id) for sorting
            synthesis_node_distances = [(raw_distances[nid], nid) for nid in reached_synthesis_nodes]
            # Sort by distance DESCENDING (furthest first)
            synthesis_node_distances.sort(key=lambda x: x[0], reverse=True)
            logger.debug(f"Ranked synthesis nodes by distance (desc): {synthesis_node_distances}")

            # --- Step 6: Assign Consecutive Levels Based on Rank --- 
            for i, (dist, node_id) in enumerate(synthesis_node_distances):
                level = i + 1
                node_levels[node_id] = level
                logger.debug(f"  Level Calculation: Node {node_id} (rank={i}, dist={dist}) -> level={level}")
            
            unreached_synthesis = synthesis_nodes_in_tree - reached_synthesis_nodes
            if unreached_synthesis:
                 logger.warning(f"The following synthesis nodes were in the tree but not reached during backward BFS: {unreached_synthesis}")
        else:
            logger.warning("No synthesis nodes were reached during backward BFS. Levels cannot be assigned.")
            

    # --- Step 7: Assemble Final Output Nodes --- 
    final_nodes_list = []
    for node_id in tree_nodes_ids:
        # Ensure node_id exists in original data (should always be true here due to Step 1 check)
        if node_id in all_nodes_data:
            node_data = copy.deepcopy(all_nodes_data[node_id])
            # Add level *only* if it's a synthesis node and a level was calculated
            if node_id in node_levels:
                 node_data['synthesis_level'] = node_levels[node_id]
            # Otherwise (chemical node or unreachable synth node), do not add the key
            
            final_nodes_list.append(node_data)
        else:
             # This case should ideally not be reached if Step 1 is correct
              logger.error(f"INTERNAL ERROR: Node ID {node_id} was in tree_nodes_ids but not in all_nodes_data.")

    # --- Step 8: Collect Final Edges --- 
    final_edges_list = [copy.deepcopy(all_edges_data[edge_id]) for edge_id in tree_edges_ids if edge_id in all_edges_data]

    # --- Step 9: Find and Add Testing Nodes and Edges --- 
    added_test_nodes = 0
    added_test_edges = 0
    if all_testing_nodes_data and all_testing_edges_data:
        logger.debug(f"Searching for testing data linked to catalyst '{target_catalyst_id}'.")
        # Iterate through all *testing* edges to find those starting from the catalyst
        for edge_id, edge_data in all_testing_edges_data.items():
            if edge_data.get('source_id') == target_catalyst_id and edge_data.get('type') == 'tested_in':
                target_test_node_id = edge_data.get('target_id')
                if not target_test_node_id:
                    logger.warning(f"Testing edge '{edge_id}' linked to catalyst '{target_catalyst_id}' is missing a target_id.")
                    continue

                # Add the testing edge
                final_edges_list.append(copy.deepcopy(edge_data))
                added_test_edges += 1
                logger.debug(f"  Added testing edge: {edge_id} ({target_catalyst_id} -> {target_test_node_id})")

                # Add the corresponding testing node if not already added (unlikely but possible)
                # Check if the target test node is already in final_nodes_list by its ID
                if not any(node['id'] == target_test_node_id for node in final_nodes_list):
                    if target_test_node_id in all_testing_nodes_data:
                        test_node_data = copy.deepcopy(all_testing_nodes_data[target_test_node_id])
                        final_nodes_list.append(test_node_data)
                        added_test_nodes += 1
                        logger.debug(f"  Added testing node: {target_test_node_id}")
                    else:
                        logger.warning(f"Testing node '{target_test_node_id}' referenced by edge '{edge_id}' was not found in all_testing_nodes_data.")
        if added_test_nodes > 0 or added_test_edges > 0:
             logger.info(f"Added {added_test_nodes} testing nodes and {added_test_edges} testing edges linked to '{target_catalyst_id}'.")
        else:
             logger.debug(f"No testing nodes or edges found directly linked to catalyst '{target_catalyst_id}' in the provided testing data.")
    else:
        logger.debug("No testing data provided, skipping step 9.")

    # --- Step 10: Find and Add Characterization Nodes and Edges ---
    added_char_nodes = 0
    added_char_edges = 0
    if all_char_nodes_data and all_char_edges_data:
        logger.debug(f"Searching for characterization data linked to catalyst '{target_catalyst_id}'.")
        for edge_id, edge_data in all_char_edges_data.items():
            if edge_data.get('source_id') == target_catalyst_id and edge_data.get('type') == 'characterized_in':
                target_char_node_id = edge_data.get('target_id')
                if not target_char_node_id:
                    logger.warning(f"Characterization edge '{edge_id}' linked to catalyst '{target_catalyst_id}' is missing a target_id.")
                    continue

                # Add the characterization edge
                final_edges_list.append(copy.deepcopy(edge_data))
                added_char_edges += 1
                logger.debug(f"  Added characterization edge: {edge_id} ({target_catalyst_id} -> {target_char_node_id})")

                # Add the corresponding characterization node if not already included
                if not any(node['id'] == target_char_node_id for node in final_nodes_list):
                    if target_char_node_id in all_char_nodes_data:
                        char_node_data = copy.deepcopy(all_char_nodes_data[target_char_node_id])
                        final_nodes_list.append(char_node_data)
                        added_char_nodes += 1
                        logger.debug(f"  Added characterization node: {target_char_node_id}")
                    else:
                        logger.warning(f"Characterization node '{target_char_node_id}' referenced by edge '{edge_id}' was not found in all_char_nodes_data.")
        if added_char_nodes > 0 or added_char_edges > 0:
            logger.info(f"Added {added_char_nodes} characterization nodes and {added_char_edges} characterization edges linked to '{target_catalyst_id}'.")
    else:
        logger.debug("No characterization data provided, skipping characterization augmentation.")


    total_nodes = len(final_nodes_list)
    total_edges = len(final_edges_list)
    logger.info(f"Extracted graph for '{target_catalyst_id}': {total_nodes} nodes ({len(tree_nodes_ids)} synthesis, {added_test_nodes} testing, {added_char_nodes} characterization), {total_edges} edges ({len(tree_edges_ids)} synthesis, {added_test_edges} testing, {added_char_edges} characterization).")

    # Restore the original return structure you added manually, but use a more general key
    return {"catgraph_tree":
        {
        "catalyst_id": target_catalyst_id,
        "nodes": final_nodes_list,
        "edges": final_edges_list
        }
    }

def process_file(input_path: Path, output_path: Path) -> List[Path]:
    """Processes a single combined graph JSON, extracts catalyst trees,
       and saves them to the output directory.

    Args:
        input_path: Path to the input *_output.json file.
        output_path: Directory to save the *_catgraph_tree.json files.

    Returns:
        A list of paths to the generated *_catgraph_tree.json files.
    """
    # Load JSON data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        logger.info(f"Successfully loaded JSON data from {input_path}")
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {input_path}")
        return
    except FileNotFoundError:
        logger.error(f"Error: Input JSON file not found: {input_path}")
        return
    except Exception as e:
        logger.error(f"Error reading JSON file {input_path}: {e}")
        return

    # Validate Synthesis section
    synthesis_data = json_data.get("synthesis")
    if not synthesis_data or "nodes" not in synthesis_data or "edges" not in synthesis_data:
        logger.error("Error: JSON data must contain a 'synthesis' section with 'nodes' and 'edges'.")
        return

    tested_catalyst_ids = synthesis_data.get("catalyst_tested_ids")
    if not tested_catalyst_ids or not isinstance(tested_catalyst_ids, list):
        logger.warning("Warning: 'synthesis' section does not contain a 'catalyst_tested_ids' list. Attempting to find catalysts marked with 'is_tested_catalyst'.")
        # Fallback: Identify tested catalysts from node properties if list is missing
        all_synth_nodes_data_temp = {node['id']: node for node in synthesis_data.get("nodes", [])}
        tested_catalyst_ids = [
            nid for nid, ndata in all_synth_nodes_data_temp.items()
            if ndata.get("is_tested_catalyst") is True and ndata.get("type") == "chemical"
        ]
        if not tested_catalyst_ids:
             logger.error("Error: Could not find 'catalyst_tested_ids' list or any nodes marked with 'is_tested_catalyst: true' in the synthesis section.")
             return
        logger.info(f"Found tested catalyst IDs from node properties: {tested_catalyst_ids}")

    # Validate Testing section
    testing_data = json_data.get("testing")
    if not testing_data or "nodes" not in testing_data or "edges" not in testing_data:
        logger.warning("Warning: JSON data does not contain a valid 'testing' section with 'nodes' and 'edges'. Output trees will only contain synthesis data.")
        all_testing_nodes_data = {}
        all_testing_edges_data = {}
    else:
        logger.info("Found 'testing' section.")
        all_testing_nodes_data = {node['id']: node for node in testing_data.get("nodes", [])}
        all_testing_edges_data = {edge['id']: edge for edge in testing_data.get("edges", []) if 'id' in edge}
        # Log edges missing IDs
        for i, edge in enumerate(testing_data.get("edges", [])):
            if 'id' not in edge:
                logger.warning(f"Testing edge at index {i} (source: {edge.get('source_id')}, target: {edge.get('target_id')}) is missing an 'id' and will be ignored.")
    
    # Validate Characterization section (optional)
    char_data = json_data.get("characterization")
    if not char_data or "nodes" not in char_data or "edges" not in char_data:
        logger.info("Info: JSON data does not contain a valid 'characterization' section with 'nodes' and 'edges'. Trees will not include characterization.")
        all_char_nodes_data = {}
        all_char_edges_data = {}
    else:
        logger.info("Found 'characterization' section.")
        all_char_nodes_data = {node['id']: node for node in char_data.get("nodes", [])}
        all_char_edges_data = {edge['id']: edge for edge in char_data.get("edges", []) if 'id' in edge}
        for i, edge in enumerate(char_data.get("edges", [])):
            if 'id' not in edge:
                logger.warning(f"Characterization edge at index {i} (source: {edge.get('source_id')}, target: {edge.get('target_id')}) is missing an 'id' and will be ignored.")

    # Build the full synthesis graph
    logger.info("Building the full synthesis graph...")
    full_graph, all_synthesis_nodes_data, all_synthesis_edges_data = build_full_synthesis_graph(synthesis_data)
    logger.info(f"Graph built with {full_graph.number_of_nodes()} nodes and {full_graph.number_of_edges()} edges.")

    # Extract tree for each tested catalyst
    num_trees_extracted = 0
    generated_tree_files = [] # List to store paths of generated files
    for catalyst_id in tested_catalyst_ids:
        logger.info(f"--- Extracting graph for catalyst: {catalyst_id} ---")
        # Pass testing data along with synthesis data
        tree_data = extract_catgraph_tree(
            full_graph,
            catalyst_id,
            all_synthesis_nodes_data,
            all_synthesis_edges_data,
            all_testing_nodes_data,
            all_testing_edges_data,
            all_char_nodes_data,
            all_char_edges_data
        )

        if tree_data:
            # Save the tree to a JSON file with updated name
            output_filename = output_path / f"{catalyst_id}_catgraph_tree.json"
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(tree_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved CatGraph tree for {catalyst_id} to {output_filename}")
                generated_tree_files.append(output_filename) # Add path to list
                num_trees_extracted += 1
            except Exception as e:
                logger.error(f"Failed to save CatGraph tree for {catalyst_id} to {output_filename}: {e}")
        else:
             logger.warning(f"Could not extract tree for catalyst ID: {catalyst_id}")

    logger.info(f"--- Extraction complete. Extracted {num_trees_extracted} CatGraph trees from {input_path}. ---")
    return generated_tree_files # Return the list of paths

def main():
    parser = argparse.ArgumentParser(description="Extract synthesis trees including testing data for tested catalysts from a CatGraphNX JSON file.")
    parser.add_argument("json_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Directory to save the output JSON tree files (default: current directory).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist

    input_path = Path(args.json_file)
    process_file(input_path, output_path)


if __name__ == "__main__":
    main() 