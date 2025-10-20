import json
import logging
import os
import re # Added for regex parsing
import io # Added for pandas
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd # Added for CSV parsing

# Import prompts - Assuming prompts.py is accessible
from CATDA.prompts.extract_prompts import ml_dataset_row_generation
# Import functions from sibling tools
from CATDA.tools.cat_graph.extract_synthesis_trees import process_file as extract_trees
from CATDA.tools.cat_graph.catgraph_to_txt import generate_catgraph_text

logger = logging.getLogger(__name__)

def load_feature_descriptions(feature_file_path: Path) -> str:
    """Loads feature descriptions from the specified file."""
    try:
        with open(feature_file_path, 'r', encoding='utf-8') as f:
            # Skip header line if present, join the rest
            lines = f.readlines()
            if lines and '|' in lines[0]: # Basic check for header
                 return "".join(lines[1:])
            else:
                 return "".join(lines)
    except FileNotFoundError:
        logger.error(f"Feature description file not found: {feature_file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading feature description file {feature_file_path}: {e}")
        raise

def generate_feature_row(scenario_text: str, model, prompt_template: str, feature_descriptions: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generates features for a given scenario using the LLM.

    Assumes the LLM returns pipe-separated data enclosed in ```csv ... ```.
    Parses the output using regex and pandas.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str]]: 
            (parsed_dataframe, raw_llm_output) or (None, raw_llm_output) on failure.
    """
    prompt = prompt_template.format(
        extract_feature_descriptions=feature_descriptions,
        scenario_text=scenario_text
    )
    raw_llm_output = None # Initialize
    try:
        response = model.invoke(prompt)
        raw_llm_output = response.content.strip()

        # 1. Extract content within ```csv block
        match = re.search(r"```csv\n(.*?)```", raw_llm_output, re.DOTALL | re.IGNORECASE)
        if not match:
            logger.warning(f"LLM output did not contain expected ```csv block. Output: {raw_llm_output}")
            return None, raw_llm_output
        
        csv_content = match.group(1).strip()
        if not csv_content:
             logger.warning(f"Extracted ```csv block is empty. Raw output: {raw_llm_output}")
             return None, raw_llm_output

        # 2. Parse using pandas
        try:
            # Use StringIO to treat the string as a file
            df = pd.read_csv(io.StringIO(csv_content), sep='|')
            
            # 3. Basic validation (check if DataFrame is empty)
            if df.empty:
                logger.warning(f"Parsed CSV is empty. Content:\n{csv_content}")
                return None, raw_llm_output
            
            # Clean column names (strip whitespace)
            df.columns = [str(col).strip() for col in df.columns]

            # Optional: Clean data in cells (strip whitespace)
            # df = df.map(lambda x: x.strip() if isinstance(x, str) else x) 
            # Commented out for now - might be too aggressive depending on data

            logger.debug(f"Successfully parsed DataFrame. Shape: {df.shape}")
            return df, raw_llm_output

        except pd.errors.ParserError as pe:
            logger.warning(f"Pandas failed to parse the extracted CSV content. Error: {pe}. Content:\n{csv_content}")
            return None, raw_llm_output
        except Exception as parse_e:
            logger.error(f"Unexpected error during pandas parsing. Error: {parse_e}. Content:\n{csv_content}", exc_info=True)
            return None, raw_llm_output

    except Exception as e:
        logger.error(f"Error invoking LLM for scenario: {e}", exc_info=True)
        # Return raw output even if LLM call failed
        return None, raw_llm_output


def generate_ml_dataset(combined_graph_file: Path, run_id: str, output_dir: Path, ml_model_name: str, ml_model: Any, feature_file_path: Path):
    """
    Orchestrates the ML dataset generation process for a SINGLE combined graph file.
    Saves results to a file specific to the run_id.

    Args:
        combined_graph_file: Path to the *_output.json file.
        run_id: Identifier for the current run (file stem).
        output_dir: The main output directory (e.g., 'output_v3').
        model_name: Name of the LLM to use for row generation.
        feature_file_path: Path to feature descriptions txt file.
    """
    logger.info(f"[{run_id}] --- Starting ML Dataset Generation for {combined_graph_file.name} ---")

    # Define subdirectories - use run_id for temp dirs
    temp_dir = output_dir / f"ml_dataset_temp_{run_id}"
    tree_output_dir = temp_dir / "trees"
    scenario_output_dir = temp_dir / "scenarios"
    final_dataset_dir = output_dir / "dataset"
    final_dataset_file = final_dataset_dir / f"{run_id}_dataset.tsv"
    # New directory for scenario-response pairs
    scenario_response_pairs_dir = output_dir / "scenario_response_pairs" / run_id

    # Create directories
    tree_output_dir.mkdir(parents=True, exist_ok=True)
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    final_dataset_dir.mkdir(parents=True, exist_ok=True)
    scenario_response_pairs_dir.mkdir(parents=True, exist_ok=True) # Create the new directory

    logger.info(f"[{run_id}] Temporary tree files will be stored in: {tree_output_dir}")
    logger.info(f"[{run_id}] Temporary scenario text files will be stored in: {scenario_output_dir}")
    logger.info(f"[{run_id}] Scenario-Response pairs will be stored in: {scenario_response_pairs_dir}")
    logger.info(f"[{run_id}] Final dataset target: {final_dataset_file}")

    all_tree_files = []
    # 1. Extract individual catalyst trees from the single combined graph
    logger.info(f"[{run_id}] Step 1: Extracting individual catalyst trees from {combined_graph_file.name}...")
    try:
        # Pass the single file path
        generated_trees = extract_trees(combined_graph_file, tree_output_dir)
        all_tree_files.extend(generated_trees)
        logger.info(f"[{run_id}] Extracted {len(generated_trees)} trees from {combined_graph_file.name}")
    except Exception as e:
        logger.error(f"[{run_id}] Failed to extract trees from {combined_graph_file.name}: {e}", exc_info=True)
        # Clean up temporary directory on failure
        if temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"[{run_id}] Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_err:
                logger.error(f"[{run_id}] Error cleaning up temp directory {temp_dir}: {cleanup_err}")
        return # Stop processing for this file

    if not all_tree_files:
        logger.warning(f"[{run_id}] No catalyst tree files were generated from {combined_graph_file.name}. Cannot generate dataset rows for this file.")
        # Clean up temporary directory as it's likely empty or only contains logs
        if temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"[{run_id}] Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_err:
                logger.error(f"[{run_id}] Error cleaning up temp directory {temp_dir}: {cleanup_err}")
        return

    # 2. Convert trees to text scenarios
    logger.info(f"[{run_id}] Step 2: Converting trees to text scenarios...")
    scenario_files = []
    for tree_file in all_tree_files:
        try:
            scenario_text = generate_catgraph_text(tree_file)
            if scenario_text:
                output_scenario_path = scenario_output_dir / f"{tree_file.stem}.txt"
                with open(output_scenario_path, 'w', encoding='utf-8') as f:
                    f.write(scenario_text)
                scenario_files.append(output_scenario_path)
                logger.debug(f"[{run_id}] Generated scenario text: {output_scenario_path}")
            else:
                logger.warning(f"[{run_id}] Failed to generate text for tree file: {tree_file}")
        except Exception as e:
            logger.error(f"[{run_id}] Error converting tree {tree_file} to text: {e}", exc_info=True)
    logger.info(f"[{run_id}] Generated {len(scenario_files)} scenario text files.")

    if not scenario_files:
        logger.warning(f"[{run_id}] No scenario text files were generated for {combined_graph_file.name}. Cannot proceed for this file.")
        # Clean up
        if temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"[{run_id}] Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_err:
                logger.error(f"[{run_id}] Error cleaning up temp directory {temp_dir}: {cleanup_err}")
        return

    # 3. Load prompts and features
    logger.info(f"[{run_id}] Step 3: Loading prompts and feature descriptions...")
    feature_file = Path(feature_file_path)
    try:
        feature_descriptions = load_feature_descriptions(feature_file)
        prompt_template = ml_dataset_row_generation # Loaded from prompts.py
        logger.info(f"[{run_id}] Prompts and features loaded successfully.")
    except Exception as e:
        logger.error(f"[{run_id}] Failed to load prompts or features from {feature_file}: {e}")
        raise

    # 5. Generate and collect features using LLM
    logger.info(f"[{run_id}] Step 5: Generating features for each scenario...")
    all_dataframes = [] # List to hold DataFrames from each scenario
    for scenario_file in scenario_files:
        scenario_text = None
        raw_llm_output_for_saving = None
        try:
            with open(scenario_file, 'r', encoding='utf-8') as f:
                scenario_text = f.read()

            logger.debug(f"[{run_id}] Processing scenario: {scenario_file.name}")
            # Call modified function - expecting DataFrame
            current_df, raw_llm_output_for_saving = generate_feature_row(
                scenario_text, ml_model, prompt_template, feature_descriptions
            )

            # --- Save Scenario-Response Pair (remains the same) ---
            if scenario_text is not None and raw_llm_output_for_saving is not None:
                pair_data = {
                    "scenario_file": str(scenario_file.name),
                    "scenario_text": scenario_text,
                    "llm_raw_output": raw_llm_output_for_saving
                }
                pair_output_file = scenario_response_pairs_dir / f"{scenario_file.stem}_pair.json"
                try:
                    with open(pair_output_file, 'w', encoding='utf-8') as pf:
                        json.dump(pair_data, pf, indent=4, ensure_ascii = False)
                    logger.debug(f"[{run_id}] Saved scenario-response pair to {pair_output_file}")
                except Exception as json_err:
                    logger.error(f"[{run_id}] Failed to save scenario-response pair to {pair_output_file}: {json_err}")
            # --- End Save Pair ---

            # Store the DataFrame if valid
            if current_df is not None and not current_df.empty:
                all_dataframes.append(current_df)
                logger.debug(f"[{run_id}] Added DataFrame from {scenario_file.name}. Shape: {current_df.shape}")
            else:
                 logger.warning(f"[{run_id}] No valid DataFrame returned by generate_feature_row for scenario: {scenario_file.name}")

        except FileNotFoundError:
             logger.error(f"[{run_id}] Scenario file not found: {scenario_file}", exc_info=True)
        except Exception as e:
            logger.error(f"[{run_id}] Error processing scenario file {scenario_file}: {e}", exc_info=True)
            # Save pair even if processing failed (remains the same)
            if scenario_text is not None and raw_llm_output_for_saving is not None:
                 pair_data = {
                     "scenario_file": str(scenario_file.name),
                     "scenario_text": scenario_text,
                     "llm_raw_output": raw_llm_output_for_saving,
                     "error_during_processing": str(e)
                 }
                 pair_output_file = scenario_response_pairs_dir / f"{scenario_file.stem}_pair_error.json"
                 try:
                     with open(pair_output_file, 'w', encoding='utf-8') as pf:
                         json.dump(pair_data, pf, indent=4)
                     logger.debug(f"[{run_id}] Saved scenario-response pair (with error) to {pair_output_file}")
                 except Exception as json_err:
                     logger.error(f"[{run_id}] Failed to save scenario-response pair (with error) to {pair_output_file}: {json_err}")

    total_rows_generated = sum(len(df) for df in all_dataframes)
    logger.info(f"[{run_id}] Attempted processing for {len(scenario_files)} scenarios. Collected {len(all_dataframes)} DataFrames with a total of {total_rows_generated} rows for {combined_graph_file.name}.")

    # 6. Concatenate DataFrames and save results to the final dataset file
    logger.info(f"[{run_id}] Step 6: Concatenating DataFrames and saving dataset to {final_dataset_file}...")
    if all_dataframes:
        try:
            # Concatenate all collected DataFrames
            final_df = pd.concat(all_dataframes, ignore_index=True)
            logger.info(f"[{run_id}] Concatenated DataFrames. Final shape: {final_df.shape}")
            
            # Save the final DataFrame to TSV
            final_df.to_csv(final_dataset_file, sep='\t', index=False, encoding='utf-8')
            logger.info(f"[{run_id}] Successfully saved final dataset with {len(final_df)} rows to {final_dataset_file}")

        except Exception as e:
            logger.error(f"[{run_id}] Failed to concatenate DataFrames or write final dataset file {final_dataset_file}: {e}", exc_info=True)
    else:
        logger.info(f"[{run_id}] No valid DataFrames were generated for {combined_graph_file.name}, final dataset file not created.")

    # 7. Clean up temporary directory for this run_id (trees and scenarios)
    if temp_dir.exists():
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"[{run_id}] Cleaned up temporary directory: {temp_dir}")
        except Exception as cleanup_err:
            logger.error(f"[{run_id}] Error cleaning up temp directory {temp_dir}: {cleanup_err}")

    logger.info(f"[{run_id}] --- ML Dataset Generation Finished for {combined_graph_file.name} ---")

# Example main for standalone testing (optional)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Add dummy paths and model name for testing if needed
    # test_graph_files = [Path("path/to/your/test_output.json")]
    # test_output_dir = Path("test_output_ml")
    # test_model = "google_gemini-2.5-flash-preview-04-17"
    # generate_ml_dataset(test_graph_files, test_output_dir, test_model)
    pass 