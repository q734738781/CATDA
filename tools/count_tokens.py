import os
import json
import ast # Use ast for safe evaluation of string literals
import sys

# --- Cost Configuration (Adjust these values based on your model/pricing) ---
# Example values for Gemini 1.5 Pro (check current pricing)
COST_PER_1M_INPUT = 2.5  # USD per 1 million input tokens
COST_PER_1M_OUTPUT = 10.00 # USD per 1 million output tokens
# ---

def count_tokens_in_directory(directory_path):
    """
    Counts the total input and output tokens by reading the 'usage_metadata'
    field from all JSON files in a directory.

    Args:
        directory_path (str): The path to the directory containing metadata JSON files.

    Returns:
        tuple: A tuple containing (total_input_tokens, total_output_tokens, files_processed_count).
               Returns (0, 0, 0) if the directory doesn't exist or has issues.
    """
    total_input_tokens = 0
    total_output_tokens = 0
    files_processed = 0
    files_skipped = 0
    missing_metadata_files = []
    parsing_error_files = []

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found: {directory_path}", file=sys.stderr)
        return 0, 0, 0

    print(f"Processing files in directory: {directory_path}")

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                usage_metadata_str = data.get('usage_metadata')
                if not usage_metadata_str:
                    # Also check for the stage1 metadata as a fallback
                    usage_metadata_str = data.get('usage_metadata_stage1')

                if usage_metadata_str and isinstance(usage_metadata_str, str):
                    # Safely parse the string representation of the dictionary
                    usage_dict = ast.literal_eval(usage_metadata_str)

                    # Iterate through the model keys within the usage dictionary
                    # (e.g., 'gemini-2.5-pro-preview-03-25')
                    # Assumes the first key found contains the relevant counts
                    found_tokens = False
                    for model_key in usage_dict:
                        model_usage = usage_dict[model_key]
                        if isinstance(model_usage, dict) and 'input_tokens' in model_usage and 'output_tokens' in model_usage:
                            input_tokens = model_usage.get('input_tokens', 0)
                            output_tokens = model_usage.get('output_tokens', 0)

                            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                                # NOTE: These counts are added PER FILE
                                total_input_tokens += input_tokens
                                total_output_tokens += output_tokens
                                found_tokens = True
                                break # Assume first valid entry is the one we need
                            else:
                                print(f"Warning: Non-integer token count found in {filename} for key '{model_key}'. Skipping add.", file=sys.stderr)

                    if found_tokens:
                         files_processed += 1
                    else:
                        print(f"Warning: Could not find valid 'input_tokens'/'output_tokens' structure within 'usage_metadata' for {filename}. Skipping file.", file=sys.stderr)
                        parsing_error_files.append(filename)
                        files_skipped += 1

                else:
                    print(f"Warning: 'usage_metadata' field missing or not a string in {filename}. Skipping file.", file=sys.stderr)
                    missing_metadata_files.append(filename)
                    files_skipped += 1

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}. Skipping file.", file=sys.stderr)
                parsing_error_files.append(filename)
                files_skipped += 1
            except (ValueError, SyntaxError) as e:
                 print(f"Warning: Could not parse 'usage_metadata' string in {filename}: {e}. Skipping file.", file=sys.stderr)
                 parsing_error_files.append(filename)
                 files_skipped += 1
            except Exception as e:
                print(f"Error processing file {filename}: {e}", file=sys.stderr)
                files_skipped += 1
        else:
            # Optional: print a message for non-JSON files if needed
            # print(f"Skipping non-JSON file: {filename}")
            pass

    print(f"\nFinished processing.")
    print(f"Files successfully processed: {files_processed}")
    print(f"Files skipped: {files_skipped}")
    if missing_metadata_files:
        print(f"  - Files missing 'usage_metadata': {len(missing_metadata_files)}") # ({', '.join(missing_metadata_files)})") # Optionally list files
    if parsing_error_files:
        print(f"  - Files with JSON/metadata parsing errors: {len(parsing_error_files)}") # ({', '.join(parsing_error_files)})") # Optionally list files

    # Return the count of processed files as well
    return total_input_tokens, total_output_tokens, files_processed

if __name__ == "__main__":
    # --- Set the target directory here ---
    metadata_directory = r"C:\Users\73473\Desktop\CATDA\SI_Data\Patent Data\Run Metadata"
    # ---

    if not metadata_directory:
        print("Error: Please set the 'metadata_directory' variable in the script.", file=sys.stderr)
    else:
        # Get counts and the number of files processed
        input_count, output_count, num_files_processed = count_tokens_in_directory(metadata_directory)

        print(f"\n--- Token Counts ---")
        print(f"Total Input Tokens Summed: {input_count:,}") # Add comma formatting
        print(f"Total Output Tokens Summed: {output_count:,}") # Add comma formatting

        # Calculate cost
        total_input_cost = (input_count / 1_000_000) * COST_PER_1M_INPUT
        total_output_cost = (output_count / 1_000_000) * COST_PER_1M_OUTPUT
        total_cost = total_input_cost + total_output_cost

        # Calculate average cost per document
        average_cost_per_doc = 0
        if num_files_processed > 0:
            average_cost_per_doc = total_cost / num_files_processed
        else:
            print("\nWarning: No files were processed successfully, cannot calculate average cost.", file=sys.stderr)

        print(f"\n--- Cost Calculation ---")
        print(f"Using Cost Params (USD per 1M tokens): Input=${COST_PER_1M_INPUT:.2f}, Output=${COST_PER_1M_OUTPUT:.2f}")
        print(f"Total Estimated Input Cost: ${total_input_cost:.4f}")
        print(f"Total Estimated Output Cost: ${total_output_cost:.4f}")
        print(f"Total Estimated Cost: ${total_cost:.4f}")
        if num_files_processed > 0:
            print(f"Average Estimated Cost per Document: ${average_cost_per_doc:.6f}")
