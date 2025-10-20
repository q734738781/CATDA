import json
import argparse
from pathlib import Path
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def format_dict_yaml(data, indent=0):
    """Helper function to format dictionaries into a YAML-like string."""
    lines = []
    indent_str = " " * indent
    sorted_keys = sorted(data.keys()) if isinstance(data, dict) else range(len(data)) # Handle lists too

    for key in sorted_keys:
        value = data[key]
        key_str = f"{key}:" if isinstance(data, dict) else "-" # Use '-' for list items

        if isinstance(value, dict):
            # Handle nested dictionaries with value/unit pairs specifically
            if 'value' in value and len(value) <= 2: # Check if it looks like a value/unit dict
                 val_str = str(value.get('value', 'N/A'))
                 unit_str = f" {value.get('unit', '')}" if value.get('unit') else ""
                 lines.append(f"{indent_str}{key_str} {val_str}{unit_str}")
            else:
                 # Otherwise, recurse for general dictionaries
                 lines.append(f"{indent_str}{key_str}")
                 lines.extend(format_dict_yaml(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{indent_str}{key_str}")
            # Check if list contains simple items or complex dicts
            if all(not isinstance(item, (dict, list)) for item in value):
                 lines.append(f"{indent_str}  - {', '.join(map(str, value))}") # Join simple items
            else:
                 lines.extend(format_dict_yaml(value, indent + 2)) # Recurse for lists of complex items
        else:
            lines.append(f"{indent_str}{key_str} {str(value)}")
    return lines

def generate_catgraph_text(input_json_path: Path) -> str | None:
    """
    Loads a catgraph_tree JSON file and generates a human-readable text summary.

    Args:
        input_json_path: Path to the input *_catgraph_tree.json file.

    Returns:
        A string containing the formatted text summary, or None if an error occurs.
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from {input_json_path}")
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {input_json_path}")
        return None
    except FileNotFoundError:
        logger.error(f"Error: Input JSON file not found: {input_json_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading JSON file {input_json_path}: {e}")
        return None

    if "catgraph_tree" not in data:
        logger.error("Error: JSON file does not contain a 'catgraph_tree' root key.")
        return None

    tree_data = data["catgraph_tree"]
    catalyst_id = tree_data.get("catalyst_id")
    nodes = tree_data.get("nodes", [])
    edges = tree_data.get("edges", [])

    if not catalyst_id or not nodes:
        logger.error("Error: JSON tree data is missing 'catalyst_id' or 'nodes'.")
        return None

    # --- Separate Nodes by Type ---
    synthesis_nodes = sorted(
        [n for n in nodes if n.get("type") == "synthesis" and "synthesis_level" in n],
        key=lambda x: x["synthesis_level"]
    )
    chemical_nodes = {n["id"]: n for n in nodes if n.get("type") == "chemical"}
    testing_nodes = [n for n in nodes if n.get("type") == "testing"]
    characterization_nodes = [n for n in nodes if n.get("type") == "characterization"]
    catalyst_node = chemical_nodes.get(catalyst_id)

    if not catalyst_node:
        logger.error(f"Error: Catalyst node with ID '{catalyst_id}' not found in chemical nodes.")
        return None

    output_lines = []

    # --- 1. Catalyst Information ---
    output_lines.append(f"Catalyst Node Name: {catalyst_id}")
    output_lines.append(f"Catalyst Name: {catalyst_node.get('name', 'N/A')}")
    if catalyst_node.get('aliases'):
        output_lines.append(f"  Aliases: {', '.join(catalyst_node['aliases'])}")
    if catalyst_node.get('chemical formula'):
        output_lines.append(f"  Formula: {catalyst_node['chemical formula']}")

    if catalyst_node.get('composition'):
        output_lines.append("  Composition:")
        output_lines.extend(format_dict_yaml(catalyst_node['composition'], indent=4))
    if catalyst_node.get('properties'):
        output_lines.append("  Properties:")
        output_lines.extend(format_dict_yaml(catalyst_node['properties'], indent=4))

    # --- 2. Synthesis Pathway ---
    output_lines.append("\nSynthesis Pathway:")
    if not synthesis_nodes:
        output_lines.append("  No synthesis steps found.")
    else:
        # Build lookup for edges by target and source
        edges_by_target = defaultdict(list)
        edges_by_source = defaultdict(list)
        for edge in edges:
            if edge.get("source_id") and edge.get("target_id"):
                edges_by_target[edge["target_id"]].append(edge)
                edges_by_source[edge["source_id"]].append(edge)

        for syn_node in synthesis_nodes:
            syn_id = syn_node["id"]
            level = syn_node.get("synthesis_level", "N/A")
            output_lines.append(f"\n  Step {level}: {syn_node.get('name', 'N/A')} (ID: {syn_id})")
            if syn_node.get('procedure'):
                 output_lines.append(f"    Procedure: {syn_node['procedure']}")

            # Find Inputs
            input_details = [] # Store tuples of (name_str, node_data)
            for edge in edges_by_target.get(syn_id, []):
                source_id = edge.get("source_id")
                if source_id in chemical_nodes:
                    chem_node = chemical_nodes[source_id]
                    chem_name = chem_node.get('name', source_id)
                    role = edge.get('properties', {}).get('role', '')
                    chem_str = chem_name
                    if role:
                        chem_str += f" ({role})"
                    input_details.append((chem_str, chem_node)) # Store name and node

            if input_details:
                 output_lines.append(f"    Inputs:")
                 # Sort by chemical name string for consistent order
                 for chem_str, chem_node in sorted(input_details, key=lambda item: item[0]):
                     output_lines.append(f"      - {chem_str}")
                     if chem_node.get('composition'):
                         output_lines.append("        Composition:")
                         output_lines.extend(format_dict_yaml(chem_node['composition'], indent=10))
                     if chem_node.get('properties'):
                         output_lines.append("        Properties:")
                         output_lines.extend(format_dict_yaml(chem_node['properties'], indent=10))

             # Find Outputs
            output_details = [] # Store tuples of (name_str, node_data)
            for edge in edges_by_source.get(syn_id, []):
                target_id = edge.get("target_id")
                if target_id in chemical_nodes:
                    chem_node = chemical_nodes[target_id]
                    chem_name = chem_node.get('name', target_id)
                    output_details.append((chem_name, chem_node)) # Store name and node

            if output_details:
                 output_lines.append(f"    Outputs:")
                 # Sort by chemical name string for consistent order
                 for chem_name, chem_node in sorted(output_details, key=lambda item: item[0]):
                     output_lines.append(f"      - {chem_name}")
                     if chem_node.get('composition'):
                         output_lines.append("        Composition:")
                         output_lines.extend(format_dict_yaml(chem_node['composition'], indent=10))
                     if chem_node.get('properties'):
                         output_lines.append("        Properties:")
                         output_lines.extend(format_dict_yaml(chem_node['properties'], indent=10))

            if syn_node.get('conditions'):
                output_lines.append("    Conditions:")
                output_lines.extend(format_dict_yaml(syn_node['conditions'], indent=6))

    # --- 3. Testing Information ---
    output_lines.append("\nTesting Scenarios:")
    if not testing_nodes:
        output_lines.append("  No testing scenarios found for this catalyst in the tree.")
    else:
        for i, test_node in enumerate(testing_nodes):
            test_id = test_node.get('id', f"Test_{i+1}")
            output_lines.append(f"\n  Scenario {i+1}: {test_node.get('description', 'N/A')} (ID: {test_id})")

            if test_node.get('conditions_json'):
                output_lines.append("    Conditions:")
                output_lines.extend(format_dict_yaml(test_node['conditions_json'], indent=6))
            elif test_node.get('conditions'): # Fallback for older format
                 output_lines.append("    Conditions (Legacy):")
                 output_lines.extend(format_dict_yaml(test_node['conditions'], indent=6))

            if test_node.get('results_json'):
                output_lines.append("    Results:")
                output_lines.extend(format_dict_yaml(test_node['results_json'], indent=6))
            elif test_node.get('results'): # Fallback for older format
                 output_lines.append("    Results (Legacy):")
                 output_lines.extend(format_dict_yaml(test_node['results'], indent=6))

    # --- 4. Characterization Information ---
    output_lines.append("\nCharacterization:")
    if not characterization_nodes:
        output_lines.append("  No characterization nodes found for this catalyst in the tree.")
    else:
        for i, char_node in enumerate(characterization_nodes):
            char_id = char_node.get('id', f"Char_{i+1}")
            method_name = char_node.get('method_name', 'N/A') or char_node.get('method', 'N/A')
            is_reported = char_node.get('data_reported')
            if is_reported is None:
                is_reported = char_node.get('is_reported')
            summary = char_node.get('characterization_summary') or char_node.get('summary')
            output_lines.append(f"\n  Characterization {i+1}: {method_name} (ID: {char_id})")
            if is_reported is not None:
                output_lines.append(f"    Data reported: {is_reported}")
            if summary:
                output_lines.append(f"    Summary: {summary}")
            evidence = char_node.get('evidence_snippet')
            if evidence:
                output_lines.append(f"    Evidence: {evidence}")

    return "\n".join(output_lines)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a human-readable text summary from a catgraph_tree JSON file."
    )
    parser.add_argument(
        "input_json", type=str, help="Path to the input *_catgraph_tree.json file."
    )
    parser.add_argument(
        "-o",
        "--output_txt",
        type=str,
        help="Optional path to save the output text summary file. If not provided, prints to console.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    input_path = Path(args.input_json)

    # Call the core logic function
    summary_text = generate_catgraph_text(input_path)

    if summary_text:
        if args.output_txt:
            output_path = Path(args.output_txt)
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                logger.info(f"Successfully saved summary to {output_path}")
            except Exception as e:
                logger.error(f"Failed to write summary to {output_path}: {e}")
        else:
            print(summary_text)

if __name__ == "__main__":
    main() 