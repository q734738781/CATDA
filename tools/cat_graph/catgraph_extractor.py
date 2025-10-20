import os
import json
import re
import json5
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from langchain_core.messages import HumanMessage, AIMessage
from CATDA.prompts.extract_prompts import (
    synthesis_graph_prompt,
    synthesis_missing_check_prompt,
    testing_graph_prompt,
    testing_missing_check_prompt,
    characterization_graph_prompt,
    characterization_missing_check_prompt,
)


# Define the specific control symbols to remove
CONTROL_SYMBOLS_TO_REMOVE = [
    "<!-- PageBreak -->",
    "☐",
    "☒",
    "\u200b",  # zero-width space
    "\ufeff"   # byte order mark
]

# Set up logger for this module
logger = logging.getLogger(__name__)

# --- Helper function to apply graph changes --- #

def _apply_graph_changes(initial_graph: Dict[str, Any], changes: Dict[str, Any], graph_type: str):
    """
    Applies changes (add, update, delete) to nodes and edges of a graph dictionary.
    Modifies the initial_graph dictionary in place.
    """
    if not initial_graph or not isinstance(initial_graph, dict):
        logger.warning(f"Cannot apply {graph_type} changes: Initial graph is invalid or missing.")
        return

    if not changes or not isinstance(changes, dict):
        logger.info(f"No valid {graph_type} changes provided or changes structure is invalid.")
        return

    # Ensure base structure exists
    if 'nodes' not in initial_graph:
        initial_graph['nodes'] = []
    if 'edges' not in initial_graph:
        initial_graph['edges'] = []

    # --- Process Deletions First --- #
    nodes_to_delete = set(changes.get('node_ids_to_delete', []))
    edges_to_delete = set(changes.get('edge_ids_to_delete', []))

    if nodes_to_delete:
        initial_graph['nodes'] = [n for n in initial_graph['nodes'] if n.get('id') not in nodes_to_delete]
        logger.info(f"Deleted {len(nodes_to_delete)} {graph_type} nodes.")

    if edges_to_delete:
        initial_graph['edges'] = [e for e in initial_graph['edges'] if e.get('id') not in edges_to_delete]
        logger.info(f"Deleted {len(edges_to_delete)} {graph_type} edges.")

    # --- Process Updates --- #
    nodes_to_update = changes.get('nodes_to_update', [])
    edges_to_update = changes.get('edges_to_update', [])

    if nodes_to_update:
        node_map = {n['id']: n for n in nodes_to_update if n.get('id')}
        updated_count = 0
        for i, node in enumerate(initial_graph['nodes']):
            node_id = node.get('id')
            if node_id in node_map:
                initial_graph['nodes'][i] = node_map[node_id]
                updated_count += 1
        logger.info(f"Updated {updated_count} {graph_type} nodes.")

    if edges_to_update:
        edge_map = {e['id']: e for e in edges_to_update if e.get('id')}
        updated_count = 0
        for i, edge in enumerate(initial_graph['edges']):
            edge_id = edge.get('id')
            if edge_id in edge_map:
                initial_graph['edges'][i] = edge_map[edge_id]
                updated_count += 1
        logger.info(f"Updated {updated_count} {graph_type} edges.")

    # --- Process Additions --- #
    nodes_to_add = changes.get('nodes_to_add', [])
    edges_to_add = changes.get('edges_to_add', [])

    if nodes_to_add:
        initial_graph['nodes'].extend(nodes_to_add)
        logger.info(f"Added {len(nodes_to_add)} {graph_type} nodes.")

    if edges_to_add:
        initial_graph['edges'].extend(edges_to_add)
        logger.info(f"Added {len(edges_to_add)} {graph_type} edges.")

# --- Main functionality ---

def extract_catgraph(file_path: Path, output_dir: Path, model, model_name: str) -> Dict:
    """
    Extracts the CatGraph from a single file using the specified model
    and tracks usage metadata. Saves the result for this file individually.
    """
    logger.info(f"Starting CatGraph extraction for: {file_path}")
    # Use a more informative run_id including timestamp
    run_id = f"{file_path.stem}"
    output_file = output_dir / "graph" / f"{run_id}_output.json" # File for combined graph

    # Create output and metadata directories if they don't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # 1. Remove Control Symbols
        text_for_llm = remove_control_symbols(raw_text)

        # 2. Prepare prompts
        prompt_initial = synthesis_graph_prompt.format(ARTICLE_TEXT=text_for_llm) # Use imported prompt
        # Prepare messages for potential second turn (validation check)
        messages_for_check = [
            HumanMessage(content=prompt_initial),
            # AIMessage will be added dynamically after the first call
            HumanMessage(content=synthesis_missing_check_prompt)
        ]

        # 3. Call LLM(s) within the usage metadata callback context
        logger.info(f"[{os.getpid()}] Running LLM calls for {file_path}...")
        # First turn: Send the initial prompt for synthesis
        response_initial = model.invoke(prompt_initial)
        llm_output_initial = response_initial.content
        # Add the AI response to the message history for the check
        messages_for_check.insert(1, AIMessage(content=llm_output_initial))

        # -- Initial Extraction Parsing (inside 'with' to associate errors) --
        try:
            json_match_initial = re.search(r"```json\n(.*?)\n```|(\{.*?\})|(\[.*?\])", llm_output_initial, re.DOTALL | re.IGNORECASE)
            if json_match_initial:
                json_str_initial = next(g for g in json_match_initial.groups() if g is not None)
                intial_synthesis_data = json5.loads(json_str_initial)
                logger.info(f"Successfully parsed initial JSON from LLM output for {file_path}")
            else:
                logger.warning(f"Could not find JSON block in initial LLM output for {file_path}. Trying direct parse.")
                try:
                    intial_synthesis_data = json5.loads(llm_output_initial)
                    logger.info(f"Successfully parsed direct initial LLM output as JSON for {file_path}")
                except ValueError:
                    logger.error(f"Failed to parse initial LLM output as JSON for {file_path}. Output:\n{llm_output_initial}")
                    raise ValueError("Initial LLM output is not valid JSON or JSON5.")
        except ValueError as e:
            logger.error(f"Initial JSON Parsing Error for {file_path}: {e}. Raw output:\n{llm_output_initial}")
            # Re-raise to be caught by the outer try-except, ensuring metadata is captured
            raise ValueError(f"Failed to parse initial LLM output: {e}") from e

        # --- Synthesis Validation & Correction Step (Conditional) --- #
        if intial_synthesis_data: # Only run if initial extraction was successful
            logger.info(f"[{os.getpid()}] Running Synthesis Validation & Correction for {file_path}...")
            response_synthesis_check = model.invoke(messages_for_check)
            llm_output_synthesis_check = response_synthesis_check.content

            try:
                match_synth_check = re.search(r"```json\n(.*?)\n```", llm_output_synthesis_check, re.DOTALL | re.IGNORECASE)
                if match_synth_check:
                    synth_check_json_str = match_synth_check.group(1)
                    synthesis_changes = json5.loads(synth_check_json_str)
                    # Apply the changes using the helper function
                    _apply_graph_changes(intial_synthesis_data, synthesis_changes, "Synthesis")
                    logger.info(f"Applied Synthesis changes for {file_path}")
                else:
                    logger.warning(f"No ```json``` block in Synthesis validation output for {file_path}. No changes applied. Output: {llm_output_synthesis_check}")
            except ValueError as e:
                logger.error(f"Synthesis Validation JSON Parsing Error: {e}. No changes applied. Output: {llm_output_synthesis_check}")
            except Exception as e_apply: # Catch errors during the apply step
                logger.error(f"Error applying Synthesis changes: {e_apply}. Changes JSON: {synthesis_changes}")
        else:
            logger.warning(f"Skipping Synthesis validation due to initial extraction failure for {file_path}.")

        # ------------------------------------------------------------------
        # 9. Run TESTING extraction using catalyst IDs from (potentially validated) synthesis stage
        # ------------------------------------------------------------------

        testing_graph_data = None # Reset before testing extraction
        # Get catalyst IDs from the potentially updated synthesis data
        catalyst_ids_json = json.dumps(intial_synthesis_data.get("catalyst_tested_ids", [])) if intial_synthesis_data else "[]"

        if intial_synthesis_data: # Proceed only if we have *some* synthesis data (initial or validated)
            # Build testing prompts
            prompt_testing_initial = testing_graph_prompt.replace("{CATALYST_IDS_FROM_SYNTHESIS}", catalyst_ids_json).format(ARTICLE_TEXT=text_for_llm)
            messages_testing_check = [
                HumanMessage(content=prompt_testing_initial),
                # AI Message for initial testing output will be inserted here
                HumanMessage(content=testing_missing_check_prompt.replace("{CATALYST_IDS_FROM_SYNTHESIS}", catalyst_ids_json))
            ]

            # Run initial testing extraction
            logger.info(f"[{os.getpid()}] Running Initial Testing Extraction for {file_path}...")
            resp_test_initial = model.invoke(prompt_testing_initial)
            test_output_initial = resp_test_initial.content
            # Insert the initial testing output for the validation step
            messages_testing_check.insert(1, AIMessage(content=test_output_initial))

            # Parse initial testing graph
            try:
                match_test_initial = re.search(r"```json\n(.*?)\n```|(\{.*?\})", test_output_initial, re.DOTALL | re.IGNORECASE)
                if match_test_initial:
                    test_json_str = next(g for g in match_test_initial.groups() if g)
                    testing_graph_data = json5.loads(test_json_str)
                    logger.info(f"Successfully parsed initial Testing JSON from LLM output for {file_path}")
                else:
                    logger.warning(f"Could not find JSON block in initial Testing LLM output for {file_path}. Trying direct parse.")
                    try:
                        testing_graph_data = json5.loads(test_output_initial)
                        logger.info(f"Successfully parsed direct initial Testing LLM output as JSON for {file_path}")
                    except ValueError:
                       logger.error(f"Failed to parse initial Testing LLM output as JSON for {file_path}. Output:\n{test_output_initial}")
                       # Set to None to prevent validation check if initial parse fails
                       testing_graph_data = None
            except Exception as e_parse_test:
                logger.error(f"Failed parsing initial testing graph JSON for {file_path}: {e_parse_test}")
                testing_graph_data = None # Ensure it's None if parsing fails

            # --- Testing Validation & Correction Step (Conditional) --- #
            if testing_graph_data: # Only run if initial testing parse was successful
                logger.info(f"[{os.getpid()}] Running Testing Validation & Correction for {file_path}...")
                response_testing_check = model.invoke(messages_testing_check)
                llm_output_testing_check = response_testing_check.content

                try:
                    match_test_check = re.search(r"```json\n(.*?)\n```", llm_output_testing_check, re.DOTALL | re.IGNORECASE)
                    if match_test_check:
                        test_check_json_str = match_test_check.group(1)
                        testing_changes = json5.loads(test_check_json_str)
                        # Apply the changes using the helper function
                        _apply_graph_changes(testing_graph_data, testing_changes, "Testing")
                        logger.info(f"Applied Testing changes for {file_path}")
                    else:
                        logger.warning(f"No ```json``` block in Testing validation output for {file_path}. No changes applied. Output: {llm_output_testing_check}")
                except ValueError as e:
                    logger.error(f"Testing Validation JSON Parsing Error: {e}. No changes applied. Output: {llm_output_testing_check}")
                except Exception as e_apply: # Catch errors during the apply step
                     logger.error(f"Error applying Testing changes: {e_apply}. Changes JSON: {testing_changes}")
            else:
                 logger.warning(f"Skipping Testing validation due to initial extraction failure for {file_path}.")

        # --- MERGE NEW TESTING-ONLY CHEMICAL NODES ---
        # After testing extraction/validation, merge any *newly* identified chemical nodes
        # (those not present in synthesis) from testing data back into the synthesis data.
        # This ensures consistency and allows downstream processes to see all catalysts.
        if intial_synthesis_data and testing_graph_data and isinstance(intial_synthesis_data, dict) and isinstance(testing_graph_data, dict):
            synthesis_nodes = intial_synthesis_data.setdefault("nodes", [])
            synthesis_node_ids = {node.get("id") for node in synthesis_nodes if node.get("id")}
            synthesis_tested_ids = set(intial_synthesis_data.setdefault("catalyst_tested_ids", []))
            newly_added_testing_catalyst_ids = set()

            testing_nodes = testing_graph_data.get("nodes", [])
            for test_node in testing_nodes:
                test_node_id = test_node.get("id")
                test_node_type = test_node.get("type")
                # Check if it's a chemical node and *not* already in synthesis nodes
                if test_node_type == "chemical" and test_node_id and test_node_id not in synthesis_node_ids:
                    # Add the new node to synthesis nodes
                    synthesis_nodes.append(test_node)
                    synthesis_node_ids.add(test_node_id) # Update the set for quick lookup
                    # If it's a catalyst (implied by being mentioned in testing), add to tested IDs
                    newly_added_testing_catalyst_ids.add(test_node_id)
                    logger.info(f"Merged new testing-only chemical node '{test_node_id}' into synthesis data.")

            # Update the catalyst_tested_ids list in synthesis data if new ones were added
            if newly_added_testing_catalyst_ids:
                updated_tested_ids = list(synthesis_tested_ids.union(newly_added_testing_catalyst_ids))
                intial_synthesis_data["catalyst_tested_ids"] = updated_tested_ids
                logger.info(f"Updated catalyst_tested_ids in synthesis data with {len(newly_added_testing_catalyst_ids)} new IDs: {newly_added_testing_catalyst_ids}")


        # Assign is_tested_catalyst attribute to each node in synthesis with the same id as the catalyst_tested_ids
        # This should happen *after* synthesis validation AND merging new testing nodes
        if intial_synthesis_data and isinstance(intial_synthesis_data, dict):
            # Use the potentially updated list of tested IDs
            tested_ids = set(intial_synthesis_data.get("catalyst_tested_ids", []))
            for node in intial_synthesis_data.get("nodes", []):
                if node.get("id") in tested_ids:
                    node["is_tested_catalyst"] = True
                # Optional: else ensure it's false or removed if not tested
                # elif "is_tested_catalyst" in node:
                #     del node["is_tested_catalyst"]

        # Save callback metadata
        # ------------------------------------------------------------------
        # 10. Run CHARACTERIZATION extraction using catalyst IDs from synthesis stage
        # ------------------------------------------------------------------

        characterization_graph_data = None
        try:
            catalyst_ids_json_for_char = json.dumps(intial_synthesis_data.get("catalyst_tested_ids", [])) if intial_synthesis_data else "[]"
            if intial_synthesis_data:
                prompt_char_initial = characterization_graph_prompt.replace("{CATALYST_IDS_FROM_SYNTHESIS}", catalyst_ids_json_for_char).format(ARTICLE_TEXT=text_for_llm)
                messages_char_check = [
                    HumanMessage(content=prompt_char_initial),
                    # AI message placeholder
                    HumanMessage(content=characterization_missing_check_prompt.replace("{CATALYST_IDS_FROM_SYNTHESIS}", catalyst_ids_json_for_char))
                ]

                logger.info(f"[{os.getpid()}] Running Initial Characterization Extraction for {file_path}...")
                resp_char_initial = model.invoke(prompt_char_initial)
                char_output_initial = resp_char_initial.content
                messages_char_check.insert(1, AIMessage(content=char_output_initial))

                # Parse initial characterization graph
                try:
                    match_char_initial = re.search(r"```json\n(.*?)\n```|(\{.*?\})", char_output_initial, re.DOTALL | re.IGNORECASE)
                    if match_char_initial:
                        char_json_str = next(g for g in match_char_initial.groups() if g)
                        characterization_graph_data = json5.loads(char_json_str)
                        logger.info(f"Successfully parsed initial Characterization JSON from LLM output for {file_path}")
                    else:
                        logger.warning(f"Could not find JSON block in initial Characterization LLM output for {file_path}. Trying direct parse.")
                        try:
                            characterization_graph_data = json5.loads(char_output_initial)
                            logger.info(f"Successfully parsed direct initial Characterization LLM output as JSON for {file_path}")
                        except ValueError:
                            logger.error(f"Failed to parse initial Characterization LLM output as JSON for {file_path}. Output:\n{char_output_initial}")
                            characterization_graph_data = None
                except Exception as e_parse_char:
                    logger.error(f"Failed parsing initial characterization graph JSON for {file_path}: {e_parse_char}")
                    characterization_graph_data = None

                # Validation & correction
                if characterization_graph_data:
                    logger.info(f"[{os.getpid()}] Running Characterization Validation & Correction for {file_path}...")
                    response_char_check = model.invoke(messages_char_check)
                    llm_output_char_check = response_char_check.content
                    try:
                        match_char_check = re.search(r"```json\n(.*?)\n```", llm_output_char_check, re.DOTALL | re.IGNORECASE)
                        if match_char_check:
                            char_check_json_str = match_char_check.group(1)
                            char_changes = json5.loads(char_check_json_str)
                            _apply_graph_changes(characterization_graph_data, char_changes, "Characterization")
                            logger.info(f"Applied Characterization changes for {file_path}")
                        else:
                            logger.warning(f"No ```json``` block in Characterization validation output for {file_path}. No changes applied. Output: {llm_output_char_check}")
                    except ValueError as e:
                        logger.error(f"Characterization Validation JSON Parsing Error: {e}. No changes applied. Output: {llm_output_char_check}")
                    except Exception as e_apply:
                        logger.error(f"Error applying Characterization changes: {e_apply}.")

                # Merge any new chemical nodes from characterization back into synthesis (do NOT modify tested list here)
                if intial_synthesis_data and characterization_graph_data and isinstance(characterization_graph_data, dict):
                    synthesis_nodes = intial_synthesis_data.setdefault("nodes", [])
                    synthesis_node_ids = {node.get("id") for node in synthesis_nodes if node.get("id")}
                    char_nodes = characterization_graph_data.get("nodes", [])
                    added_chem_from_char = 0
                    for n in char_nodes:
                        if n.get("type") == "chemical" and n.get("id") and n["id"] not in synthesis_node_ids:
                            synthesis_nodes.append(n)
                            synthesis_node_ids.add(n["id"])
                            added_chem_from_char += 1
                    if added_chem_from_char:
                        logger.info(f"Merged {added_chem_from_char} new chemical nodes from Characterization into synthesis data.")

        except Exception as e_char:
            logger.error(f"Unhandled error during Characterization extraction for {file_path}: {e_char}", exc_info=True)

        # ------------------------------------------------------------------
        # 11. Combine (potentially validated) synthesis + testing + characterization graphs into final graph object
        # ------------------------------------------------------------------

        combined_graph = {}
        if intial_synthesis_data:
            combined_graph["synthesis"] = intial_synthesis_data
        if testing_graph_data:
            combined_graph["testing"] = testing_graph_data
        if characterization_graph_data:
            combined_graph["characterization"] = characterization_graph_data

        # 11. Save combined graph
        saved_graph_path = None
        if combined_graph: # Save only if we have at least one part (synthesis or testing)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(combined_graph, f, indent=2, ensure_ascii=False)
            saved_graph_path = str(output_file)
            logger.info(f"Saved combined graph JSON to {output_file}")
        else:
            # This case should be less likely now unless both initial extractions fail
            logger.warning(f"No graph data extracted (neither synthesis nor testing succeeded); nothing saved for {file_path}.")

        # 12. Prepare result dictionary
        final_status = 'success' # Assume success initially
        if not combined_graph:
            final_status = 'error_no_data_extracted'
        elif not intial_synthesis_data:
            final_status = 'warning_no_synthesis_data'
        elif not testing_graph_data and (intial_synthesis_data and intial_synthesis_data.get("catalyst_tested_ids")):
            # If catalysts were identified for testing, but no testing data extracted
            final_status = 'warning_no_testing_data_expected'
        elif not testing_graph_data:
            final_status = 'warning_no_testing_data'
        # Note: Characterization is optional; we do not downgrade status if missing

        result_data = {
            'file': str(file_path),
            'status': final_status,
            'run_id': run_id,
            'output_graph_file': saved_graph_path,
            'model_name': model_name, # Use model_name argument
        }

        return result_data # Return the dictionary primarily for logging in main process

    except ValueError as val_err:
        # Handle JSON parsing errors specifically (likely from initial parse)
        logger.error(f"ValueError (likely JSON parsing) during processing {file_path}: {val_err}")
        result_data = {
            'file': str(file_path),
            'status': 'error_parsing',
            'error_message': f"ValueError: {val_err}",
            'run_id': run_id,
            'model_name': model_name, # Use model_name argument
        }

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
        result_data = {
            'file': str(file_path),
            'status': 'error_unknown',
            'error_message': str(e),
            'run_id': run_id,
            'model_name': model_name, # Use model_name argument
        }

# --- Utility Functions ---

def remove_control_symbols(text: str) -> str:
    """
    Removes specific predefined control symbols from the text.
    Returns the cleaned text.
    """
    cleaned_text = text
    for symbol in CONTROL_SYMBOLS_TO_REMOVE:
        cleaned_text = cleaned_text.replace(symbol, '')
    # Additionally, strip leading/trailing whitespace that might remain
    return cleaned_text.strip()
