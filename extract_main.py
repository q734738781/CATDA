import os
import json
import logging
import argparse
import multiprocessing
from pathlib import Path
from typing import Dict, Tuple, List
from pathlib import Path as _PathAlias  # avoid confusion in type hints for argparse default
import datetime
import glob
from models.models import get_model
from CATDA.tools.ml_dataset.generate_dataset import generate_ml_dataset
from CATDA.tools.cat_graph.catgraph_extractor import extract_catgraph
from langchain_core.callbacks.usage import get_usage_metadata_callback
from langchain.globals import set_verbose
from random import shuffle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Suppress verbose logging from underlying libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

# --- Constants ---
# Model name: Provider_official-model-name. Currently support: deepseek/google/openai
# Need to configure DEEPSEEK_API_KEY/GOOGLE_API_KEY/OPENAI_API_KEY in environment variable for corresponding model
# Recommend models: openai_gpt-o1, deepseek_deepseek-reasoner (Too slow!!), google_gemini-2.5-pro(One of Best model!)
#MODEL_NAME = 'google_gemini-2.5-flash-preview-04-17' # Default model for extraction
MODEL_NAME = 'google_gemini-2.5-pro'
#ML_DATASET_MODEL_NAME = 'google_gemini-2.5-flash-preview-04-17'
ML_DATASET_MODEL_NAME = 'google_gemini-2.5-pro'
MODES = ['extract', 'generate-ml-only', 'both'] # Define available modes
ENABLE_PRINT = True
MODEL_TEMP = None # temperature for model. None for default setting

# --- Worker Function for Multiprocessing ---
def process_file_wrapper(args_tuple: Tuple[str, str, str, str, bool, str]) -> Dict:
    """
    Wrapper function to process a single file in a separate process.
    Initializes the model within the process.
    Optionally triggers ML dataset generation for the processed file.
    """
    file_path_str, output_dir_str, model_name, ml_model_name, gen_ml_flag, feature_file_str = args_tuple
    file_path = Path(file_path_str)
    output_dir = Path(output_dir_str)
    feature_file_path = Path(feature_file_str) if feature_file_str else None
    pid = os.getpid()
    logger.info(f"[{pid}] Worker started for file: {file_path}")
    if ENABLE_PRINT:
        set_verbose(True)

    if MODEL_TEMP is not None:
        model = get_model(model = model_name, temperature = MODEL_TEMP) # Initialize model with temperature
        ml_model = get_model(model = ml_model_name, temperature = MODEL_TEMP) # Initialize ML model with temperature
    else:
        model = get_model(model = model_name) # Initialize model without temperature
        ml_model = get_model(model = ml_model_name) # Initialize ML model without temperature
    result_data = None
    result_file = Path(output_dir) / "metadata" /  f"{file_path.stem}_result.json" # Consistent naming for result files
    # Create metadata directory if it doesn't exist
    result_file.parent.mkdir(parents=True, exist_ok=True)

    with get_usage_metadata_callback() as usage_cb:
        try:
            # Initialize model within the worker process
            # Using temp=0 as default, adjust if needed
            # Call the main extraction function
            # model.invoke("What is the capital of France?")
            
            result = extract_catgraph(file_path, output_dir, model, model_name)
            result['usage_metadata_stage1'] = str(usage_cb.usage_metadata)

            # --- Conditionally Generate ML Dataset Rows ---
            if gen_ml_flag and result['status'] == 'success' and result.get('output_graph_file'):
                graph_file_path = Path(result['output_graph_file'])
                run_id_for_ml = result.get('run_id', file_path.stem) # Use run_id from extraction result

                logger.info(f"[{pid}/{run_id_for_ml}] Triggering ML dataset generation for {graph_file_path.name}...")
                result['ml_generation_model'] = ml_model_name
                try:
                     # Import the generation function within the worker
                    generate_ml_dataset(graph_file_path, run_id_for_ml, output_dir, ml_model_name, ml_model, feature_file_path)
                    logger.info(f"[{pid}/{run_id_for_ml}] ML dataset generation step completed for {graph_file_path.name}.")
                    result['ml_generation_status'] = 'success'
                    result['usage_metadata'] = str(usage_cb.usage_metadata)
                except ImportError as ie:
                    logger.error(f"[{pid}/{run_id_for_ml}] Could not import ML dataset generation tools. Ensure they are correctly placed: {ie}")
                    result['ml_generation_status'] = 'error_import'
                    result['usage_metadata'] = str(usage_cb.usage_metadata)
                except Exception as ml_err:
                    logger.error(f"[{pid}/{run_id_for_ml}] Error during ML dataset generation for {graph_file_path.name}: {ml_err}", exc_info=True)
                    result['ml_generation_status'] = 'error_runtime'
                    result['usage_metadata'] = str(usage_cb.usage_metadata)
            elif gen_ml_flag:
                 logger.warning(f"[{pid}] Skipping ML dataset generation for {file_path.name} due to extraction status '{result['status']}' or missing graph file.")
                 result['ml_generation_status'] = 'skipped'
            else:
                result['ml_generation_status'] = 'not_requested'

            logger.info(f"[{pid}] Worker finished for file: {file_path} with status: {result['status']}")
            result_data = result

        except Exception as e:
            logger.error(f"[{pid}] Unhandled error in worker for {file_path}: {e}", exc_info=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_id_on_error = f"{file_path.stem}_worker_error_{timestamp}_{pid}"
            result_file = output_dir / "metadata" / f"{run_id_on_error}_result.json" # Consistent naming for error files
            result_data = {
                'file': str(file_path),
                'status': 'worker_error',
                'error_message': str(e),
                'run_id': run_id_on_error,
                'usage_metadata': str(usage_cb.usage_metadata), # document the usage data at the exception
                'model_name': model_name, # Log the intended model
            }
        # Attempt to save worker error result

        finally:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            logger.info(f"[{pid}] Saved worker result metadata to: {result_file}")

    return result_data

# --- Simplified Worker Function for ML Generation Only Mode (from Graph File) ---

def process_ml_generation_from_graph(args_tuple: Tuple[str, str, str, str]) -> Dict:
    """
    Worker function to generate ML dataset directly from a graph JSON file.
    Derives run_id from the graph filename.
    """
    graph_file_path_str, output_dir_str, ml_model_name, feature_file_str = args_tuple
    graph_file_path = Path(graph_file_path_str)
    output_dir = Path(output_dir_str) # Still potentially needed by generate_ml_dataset
    feature_file_path = Path(feature_file_str) if feature_file_str else None
    pid = os.getpid()
    logger.info(f"[{pid}] ML-Gen worker started for graph file: {graph_file_path.name}")
    if ENABLE_PRINT:
        set_verbose(True)
    # Derive run_id from filename (e.g., "my_run_id_output.json" -> "my_run_id")
    run_id_for_ml = graph_file_path.stem.replace('_output', '')

    ml_result = {
        'graph_file': str(graph_file_path),
        'ml_generation_status': 'unknown', # Default status
        'run_id': run_id_for_ml,
        'error_message': None
    }
    if MODEL_TEMP is not None:
        ml_model = get_model(model=ml_model_name, temperature=MODEL_TEMP) # Initialize ML model with temperature
    else:
        ml_model = get_model(model=ml_model_name) # Initialize ML model without temperature
    with get_usage_metadata_callback() as usage_cb:
        try:
            logger.info(f"[{pid}/{run_id_for_ml}] Triggering ML dataset generation for graph {graph_file_path.name}...")
            ml_result['ml_generation_model'] = ml_model_name
            try:
                # Import the generation function within the worker
                generate_ml_dataset(graph_file_path, run_id_for_ml, output_dir, ml_model_name, ml_model, feature_file_path)
                logger.info(f"[{pid}/{run_id_for_ml}] ML dataset generation step completed for {graph_file_path.name}.")
                ml_result['ml_generation_status'] = 'success'
                ml_result['usage_metadata'] = usage_cb.usage_metadata
            except ImportError as ie:
                logger.error(f"[{pid}/{run_id_for_ml}] Could not import ML dataset generation tools: {ie}")
                ml_result['ml_generation_status'] = 'error_import'
                ml_result['usage_metadata'] = usage_cb.usage_metadata
                ml_result['error_message'] = str(ie)
            except FileNotFoundError:
                # Explicitly catch if the graph file itself isn't found (though it should exist if passed here)
                logger.error(f"[{pid}/{run_id_for_ml}] Graph file not found during generation call: {graph_file_path}")
                ml_result['ml_generation_status'] = 'error_graph_file_not_found'
                ml_result['usage_metadata'] = usage_cb.usage_metadata
                ml_result['error_message'] = f"Graph file not found: {graph_file_path}"
            except Exception as ml_err:
                logger.error(f"[{pid}/{run_id_for_ml}] Error during ML dataset generation for {graph_file_path.name}: {ml_err}", exc_info=True)
                ml_result['ml_generation_status'] = 'error_runtime'
                ml_result['usage_metadata'] = usage_cb.usage_metadata
                ml_result['error_message'] = str(ml_err)

        except Exception as e:
            # Catch-all for unexpected errors in the worker setup itself
            logger.error(f"[{pid}] Unhandled error in ML-Gen worker for {graph_file_path}: {e}", exc_info=True)
            ml_result['ml_generation_status'] = 'worker_error'
            ml_result['usage_metadata'] = usage_cb.usage_metadata
            ml_result['error_message'] = str(e)

    logger.info(f"[{pid}] ML-Gen worker finished for: {graph_file_path.name} with status: {ml_result['ml_generation_status']}")
    # Save the result to the metadata directory
    ml_result_file = output_dir / "metadata" / f"{graph_file_path.stem}_ml_result.json"
    with open(ml_result_file, 'w', encoding='utf-8') as f:
        json.dump(ml_result, f, indent=2, ensure_ascii=False)
    logger.info(f"[{pid}] Saved ML-Gen result metadata to: {ml_result_file}")
    return ml_result


# --- Main Execution Logic ---

def main():
    # global ml_dataset_lock # Remove global declaration

    parser = argparse.ArgumentParser(description='Extract CatGraph data from text files.') # Updated description slightly
    parser.add_argument('input_path', type=str, help='Path to input file or directory (used for modes: extract, both).') # Clarified help
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save outputs and read results from (for generate-ml-only mode).') # Clarified help
    parser.add_argument('--processes', type=int, default = 4, help='Number of worker processes to use.')
    # Add other arguments as needed (e.g., file extension filter)
    parser.add_argument('--file-ext', type=str, default='.md', help='File extension to process (e.g., .txt, .md, used for modes: extract, both)') # Clarified help
    # Add mode argument
    parser.add_argument('--mode', type=str, default='both', choices=MODES, help=f'Operation mode: {MODES}. "extract" performs extraction only. "generate-ml-only" generates ML data from existing graph outputs. "both" extracts and generates ML data.') # Updated help string
    # Add argument for result file pattern (for generate-ml-only mode)
    parser.add_argument('--graph-pattern', type=str, default='*_output.json', help='Glob pattern to find graph JSON files (*_output.json) in the output_dir/graph subdirectory (used with --mode generate-ml-only).') # New argument
    # Feature description file for ML dataset generation
    parser.add_argument('--feature-file', type=str, default='./prompts/features_to_extract.txt', help='Path to feature descriptions txt used for ML dataset generation.')


    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to: {output_dir}")

    # --- Determine Run Mode ---
    run_mode = args.mode
    # generate_ml_data = args.gen_ml_dataset or run_mode == 'both' or run_mode == 'generate-ml-only' # Old logic
    generate_ml_data = run_mode == 'both' or run_mode == 'generate-ml-only' # Simplified logic

    logger.info(f"Running in mode: {run_mode}")
    if run_mode == 'both':
         logger.info(f"ML Dataset Generation is ENABLED (using model: {ML_DATASET_MODEL_NAME}).")
    elif run_mode == 'generate-ml-only':
         logger.info(f"ML Dataset Generation ONLY mode (using model: {ML_DATASET_MODEL_NAME}).")

    if run_mode == 'extract' or run_mode == 'both':
        target_worker_function = process_file_wrapper
        files_to_process = []
        if input_path.is_file():
            if input_path.suffix.lower() == args.file_ext:
                files_to_process.append(input_path)
            else:
                 logger.warning(f"Input file {input_path} does not match extension {args.file_ext}. Skipping.")
        elif input_path.is_dir():
            logger.info(f"Scanning directory {input_path} for files ending with '{args.file_ext}'...")
            files_to_process = list(input_path.rglob(f"*{args.file_ext}"))
        else:
            logger.error(f"Input path {input_path} is not a valid file or directory for mode '{run_mode}'.")
            return

        if not files_to_process:
            logger.warning(f"No input files found with extension '{args.file_ext}' in {input_path}. Exiting.")
            return

        logger.info(f"Found {len(files_to_process)} files for extraction.")

        # Prepare tasks for extraction worker
        tasks = [
            (
                str(fp),
                str(output_dir),
                MODEL_NAME,
                ML_DATASET_MODEL_NAME,
                generate_ml_data, # Pass the determined flag (True for 'both', False for 'extract')
                str(Path(args.feature_file).resolve()) if args.feature_file else ''
             ) for fp in files_to_process
        ]
        num_tasks = len(files_to_process)
        # Shuffle the tasks
        shuffle(tasks)
    elif run_mode == 'generate-ml-only':
        target_worker_function = process_ml_generation_from_graph # Use the new worker
        graph_dir = output_dir / "graph" # Define the specific subdirectory for graphs
        logger.info(f"Scanning directory '{graph_dir}' for graph files matching pattern '{args.graph_pattern}'...")
        graph_files_to_process = list(graph_dir.glob(args.graph_pattern))

        if not graph_files_to_process:
            logger.warning(f"No graph files found matching pattern '{args.graph_pattern}' in {graph_dir}. Exiting.")
            return

        logger.info(f"Found {len(graph_files_to_process)} graph files for ML dataset generation.")

        # Prepare tasks for ML generation worker (pass graph path)
        tasks = [
            (
                str(gfp), # Pass the graph file path
                str(output_dir),
                ML_DATASET_MODEL_NAME,
                str(Path(args.feature_file).resolve()) if args.feature_file else ''
            ) for gfp in graph_files_to_process
        ]
        num_tasks = len(graph_files_to_process)
        # Shuffle the tasks
        shuffle(tasks)
    else:
        logger.error(f"Invalid run mode '{run_mode}'. Should not happen due to argparse choices.")
        return

    if not tasks:
        logger.warning("No tasks generated based on the mode and inputs. Exiting.")
        return

    # --- Parallel Processing ---
    num_workers = min(args.processes, num_tasks) # Adjust workers based on actual task count
    logger.info(f"Initializing multiprocessing pool with {num_workers} workers for {num_tasks} tasks.")

    # --- Initialize Result Tracking ---
    results_summary = [] # Store basic info for final console summary
    total_tasks_processed = 0
    # Counters specific to 'extract'/'both' mode
    extraction_success_count = 0
    extraction_error_counts = { status: 0 for status in ['error_initial_extraction', 'error_parsing', 'worker_error', 'error_unknown', 'other'] }
    files_with_missing_found = 0
    # Counters specific to ML generation (relevant for all modes where it runs)
    ml_generation_success_count = 0
    ml_generation_error_counts = { status: 0 for status in ['error_import', 'error_runtime', 'error_graph_file_not_found', 'worker_error', 'other'] } # Updated ML specific errors
    ml_skipped_count = 0 # Count skipped explicitly (only relevant for extract/both mode)

    # Record start time
    start_time = datetime.datetime.now()

    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for potentially better memory usage with many tasks
        for result in pool.imap_unordered(target_worker_function, tasks):
            total_tasks_processed += 1

            if run_mode == 'extract' or run_mode == 'both':
                status = result.get('status', 'other') # Default to 'other' if status missing
                results_summary.append({ # Append minimal info for console summary
                     'file': result.get('file', 'unknown_file'),
                     'status': status,
                     'run_id': result.get('run_id', 'unknown_run_id'),
                     'ml_generation_status': result.get('ml_generation_status', 'unknown')
                })
                if status == 'success':
                    extraction_success_count += 1
                    if result.get('missing_items_found_in_check', False):
                        files_with_missing_found += 1
                    # Track ML success/skip/error *within* successful extractions
                    ml_status = result.get('ml_generation_status', 'unknown')
                    if ml_status == 'success':
                        ml_generation_success_count += 1
                    elif ml_status.startswith('skipped'):
                        ml_skipped_count += 1
                    elif ml_status != 'not_requested': # Count errors
                        error_key = ml_status if ml_status in ml_generation_error_counts else 'other'
                        ml_generation_error_counts[error_key] += 1
                elif status in extraction_error_counts:
                    extraction_error_counts[status] += 1
                else:
                    extraction_error_counts['other'] += 1 # Increment 'other' for unexpected statuses
                logger.info(f"Received result for: {result.get('file', 'unknown file')} with status: {status}, ML status: {result.get('ml_generation_status', 'N/A')}")

            elif run_mode == 'generate-ml-only':
                ml_status = result.get('ml_generation_status', 'other')
                results_summary.append({ # Append minimal info for console summary
                    'graph_file': result.get('graph_file', 'unknown_graph_file'), # Use graph_file key
                    'ml_generation_status': ml_status,
                    'run_id': result.get('run_id', 'unknown_run_id')
                })
                if ml_status == 'success':
                    ml_generation_success_count += 1
                else: # Count errors
                    error_key = ml_status if ml_status in ml_generation_error_counts else 'other'
                    ml_generation_error_counts[error_key] += 1
                logger.info(f"Received ML-Gen result for: {result.get('graph_file', 'unknown file')} with status: {ml_status}")


    # Record end time
    end_time = datetime.datetime.now()
    processing_duration = end_time - start_time

    # --- Updated Console Summary ---
    logger.info(f"Processing finished in mode: {run_mode}. See summary below.")

    print(f"\n--- CatGraph Run Summary (Mode: {run_mode}) ---")
    print(f"Processing Start Time: {start_time.isoformat()}")
    print(f"Processing End Time:   {end_time.isoformat()}")
    print(f"Processing Duration:   {processing_duration}")
    if run_mode == 'extract' or run_mode == 'both':
        print(f"Input Path: {args.input_path} (using extension: {args.file_ext})")
        print(f"Extraction Model Used: {MODEL_NAME}")
    else: # generate-ml-only
         print(f"Input: Graph files from '{args.output_dir}/graph' matching '{args.graph_pattern}'") # Updated input description
    print(f"Output Directory: {args.output_dir}")
    print(f"Workers Used: {num_workers}")
    print(f"Tasks Found: {num_tasks}")
    print(f"Tasks Processed: {total_tasks_processed}")

    if run_mode == 'extract' or run_mode == 'both':
        failed_extractions = total_tasks_processed - extraction_success_count
        print("\n-- Extraction Results --")
        print(f"Successful Extractions: {extraction_success_count}")
        print(f"Failed Extractions (Total): {failed_extractions}")
        print(f"  - Initial Extraction Errors: {extraction_error_counts['error_initial_extraction']}")
        print(f"  - JSON Parsing Errors: {extraction_error_counts['error_parsing']}")
        print(f"  - Worker Errors: {extraction_error_counts['worker_error']}")
        print(f"  - Unknown Errors: {extraction_error_counts['error_unknown']}")
        print(f"  - Other Status: {extraction_error_counts['other']}")
        print(f"Files with Missing Items Found by Check: {files_with_missing_found}")

    if generate_ml_data:
        print("\n-- ML Dataset Generation Results --")
        print(f"ML Model Used: {ML_DATASET_MODEL_NAME}")
        ml_failed_count = sum(ml_generation_error_counts.values())
        print(f"Successful ML Generations: {ml_generation_success_count}")
        # print(f"Skipped ML Generations: {ml_skipped_count}") # Skipped count is not relevant for generate-ml-only mode
        print(f"Failed ML Generations (Total): {ml_failed_count}")
        # Print specific ML errors if any occurred
        for error_type, count in ml_generation_error_counts.items():
             if count > 0:
                 print(f"  - {error_type.replace('_', ' ').title()}: {count}")


    if run_mode == 'extract' or run_mode == 'both':
         print(f"\nDetailed extraction results saved in individual JSON files within: {args.output_dir}")
    elif run_mode == 'generate-ml-only':
         print(f"\nML dataset generation attempted based on graph files in: {args.output_dir}/graph") # Updated output description
         print(f"Check logs and potential output files from 'generate_ml_dataset' for details.")


if __name__ == "__main__":
    # Required for multiprocessing freeze support on Windows
    main()
