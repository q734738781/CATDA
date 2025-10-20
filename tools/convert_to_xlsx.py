import pandas as pd
import os
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_tsv_to_xlsx(input_folder: str):
    """
    Converts all TSV files in the specified folder to XLSX format.

    Args:
        input_folder: Path to the folder containing TSV files.
    """
    logging.info(f"Scanning folder: {input_folder}")
    files_processed = 0
    files_skipped = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".tsv"):
            tsv_filepath = os.path.join(input_folder, filename)
            xlsx_filename = os.path.splitext(filename)[0] + ".xlsx"
            xlsx_filepath = os.path.join(input_folder, xlsx_filename)

            logging.info(f"Converting '{filename}' to '{xlsx_filename}'...")

            try:
                # Read TSV file
                df = pd.read_csv(tsv_filepath, sep='\t', header=0) # Assuming first row is header

                # Write to XLSX file
                # Do not overwrite
                if os.path.exists(xlsx_filepath):
                    logging.warning(f"File '{xlsx_filename}' already exists. Skipping conversion.")
                    files_skipped += 1
                    continue
                df.to_excel(xlsx_filepath, index=False)
                logging.info(f"Successfully converted '{filename}' to '{xlsx_filename}'.")
                files_processed += 1

            except pd.errors.EmptyDataError:
                logging.warning(f"Skipping empty file: '{filename}'")
                files_skipped += 1
            except Exception as e:
                logging.error(f"Error converting file '{filename}': {e}")
                files_skipped += 1
        else:
            # Optional: Log files that are not TSV if needed
            # logging.debug(f"Skipping non-tsv file: {filename}")
            pass

    logging.info(f"Conversion complete. Processed: {files_processed}, Skipped: {files_skipped}")

def main():
    parser = argparse.ArgumentParser(description="Convert TSV files to XLSX format without default styling.")
    parser.add_argument("input_folder", help="Path to the folder containing TSV files.")

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        logging.error(f"Error: Input folder not found or is not a directory: {args.input_folder}")
        return

    convert_tsv_to_xlsx(args.input_folder)

if __name__ == "__main__":
    main()
