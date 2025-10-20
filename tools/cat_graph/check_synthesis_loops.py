import json5 as json
import argparse
import logging
import os
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_synthesis_graph(json_data: dict) -> defaultdict[str, list[str]]:
    """Builds an adjacency list representation of the synthesis graph."""
    graph = defaultdict(list)
    synthesis_section = json_data.get("synthesis", {})
    nodes = {node['id'] for node in synthesis_section.get("nodes", []) if node.get('type') == 'synthesis'}
    edges = synthesis_section.get("edges", [])

    # Add nodes to the graph dictionary keys to represent all synthesis nodes, even disconnected ones
    for node_id in nodes:
        graph[node_id] # Ensures the key exists

    for edge in edges:
        source = edge.get("source_id")
        target = edge.get("target_id")
        # We only care about edges connecting two synthesis steps for cycle detection
        # Edges involving 'chemical' nodes don't form synthesis process loops
        if source in nodes and target in nodes:
            graph[source].append(target)
            logger.debug(f"Added edge: {source} -> {target}")

    return graph

def has_cycle_util(node: str, graph: defaultdict[str, list[str]], visited: set[str], recursion_stack: set[str], path: list[str]) -> tuple[bool, list[str]]:
    """Recursive utility function for DFS cycle detection."""
    visited.add(node)
    recursion_stack.add(node)
    path.append(node)
    logger.debug(f"Visiting node: {node}, Path: {' -> '.join(path)}")

    for neighbor in graph[node]:
        if neighbor not in visited:
            found_cycle, cycle_path = has_cycle_util(neighbor, graph, visited, recursion_stack, path)
            if found_cycle:
                return True, cycle_path
        elif neighbor in recursion_stack:
            # Cycle detected
            cycle_start_index = path.index(neighbor)
            cycle_path = path[cycle_start_index:] + [neighbor] # Add neighbor again to show the loop closure
            logger.info(f"Cycle detected: {' -> '.join(cycle_path)}")
            return True, cycle_path

    path.pop() # Backtrack
    recursion_stack.remove(node)
    # logger.debug(f"Finished node: {node}, Recursion Stack: {recursion_stack}") # Optional: log when backtracking
    return False, []

def detect_synthesis_cycle(graph: defaultdict[str, list[str]]) -> tuple[bool, list[str]]:
    """Detects cycles in the synthesis graph using DFS."""
    visited = set()
    recursion_stack = set()
    nodes = list(graph.keys()) # Get all synthesis nodes

    for node in nodes:
        if node not in visited:
            logger.debug(f"Starting DFS from node: {node}")
            found_cycle, cycle_path = has_cycle_util(node, graph, visited, recursion_stack, [])
            if found_cycle:
                return True, cycle_path

    return False, []

def process_file(json_file_path: str):
    """Loads a JSON file, builds the graph, and checks for cycles."""
    logger.info(f"Processing file: {json_file_path}")
    # Load JSON data
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        logger.info(f"Successfully loaded JSON data from {json_file_path}")
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {json_file_path}")
        return
    except FileNotFoundError:
        logger.error(f"Error: Input JSON file not found: {json_file_path}")
        return
    except Exception as e:
        logger.error(f"Error reading JSON file {json_file_path}: {e}")
        return

    # Basic validation
    if "synthesis" not in json_data or "nodes" not in json_data["synthesis"] or "edges" not in json_data["synthesis"]:
        logger.error(f"Error: JSON data in {json_file_path} must contain a 'synthesis' section with 'nodes' and 'edges'.")
        return

    # Build graph and detect cycles
    synthesis_graph = build_synthesis_graph(json_data)
    logger.info(f"Built synthesis graph with {len(synthesis_graph)} nodes for {json_file_path}.")
    logger.debug(f"Graph adjacency list for {json_file_path}: {dict(synthesis_graph)}")

    has_cycle, cycle_path = detect_synthesis_cycle(synthesis_graph)

    if has_cycle:
        logger.warning(f"Cycle detected in the synthesis graph for {json_file_path}!")
        print(f"File: {json_file_path} - Cycle found: {' -> '.join(cycle_path)}")
    else:
        logger.info(f"No cycles detected in the synthesis graph for {json_file_path}.")
        print(f"File: {json_file_path} - No cycles found.")
    print("-" * 30) # Separator for clarity

def main():
    parser = argparse.ArgumentParser(description="Check for cycles in the synthesis part of CatGraphNX JSON files within a directory.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing JSON files.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    input_directory = args.input_dir
    if not os.path.isdir(input_directory):
        logger.error(f"Error: Input path is not a valid directory: {input_directory}")
        return

    logger.info(f"Scanning directory: {input_directory}")
    found_files = False
    for filename in os.listdir(input_directory):
        if filename.lower().endswith((".json", ".json5")):
            found_files = True
            file_path = os.path.join(input_directory, filename)
            process_file(file_path) # Process each JSON/JSON5 file

    if not found_files:
        logger.warning(f"No .json or .json5 files found in directory: {input_directory}")

if __name__ == "__main__":
    main() 