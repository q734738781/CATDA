import os
import json
import shutil
from pathlib import Path

json_metadata_dir = r"D:\python_projects\GPT_Paper\extract_results\CATDA\all_p2_(failed)\metadata"
patent_dir = r"D:\\python_projects\\GPT_Paper\\data\\all_patent"
export_dir = r"D:\\python_projects\\GPT_Paper\\data\\all_patent_failed"

os.makedirs(export_dir, exist_ok=True)

failed_count = 0
processed_count = 0

for filename in os.listdir(json_metadata_dir):
    if filename.endswith(".json"):
        processed_count += 1
        json_path = os.path.join(json_metadata_dir, filename)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            if metadata.get("status") == "worker_error":
                relative_file_path = metadata.get("file")
                if relative_file_path:
                    # Extract the base filename from the relative path
                    base_filename = Path(relative_file_path).name
                    
                    source_file_path = os.path.join(patent_dir, base_filename)
                    destination_file_path = os.path.join(export_dir, base_filename)

                    if os.path.exists(source_file_path):
                        print(f"Copying failed file: {base_filename}")
                        #shutil.copy2(source_file_path, destination_file_path)
                        failed_count += 1
                    else:
                        print(f"Warning: Source file not found: {source_file_path}")
                else:
                    print(f"Warning: 'file' key not found or empty in {filename}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {filename}")
        except Exception as e:
            print(f"An error occurred processing file {filename}: {e}")

print(f"\nProcessed {processed_count} metadata files.")
print(f"Copied {failed_count} failed files to {export_dir}")

