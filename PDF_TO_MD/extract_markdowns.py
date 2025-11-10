import os
import shutil
from pathlib import Path

def copy_and_rename_files(root_dir: str, output_dir: str):
    """
    Copy txt files from root_dir/patent_number/txt/output.txt to output_dir/patent_number.md
    
    Args:
        root_dir: Root directory containing patent folders
        output_dir: Output directory to copy renamed files to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all subdirectories in root
    for patent_dir in os.listdir(root_dir):
        patent_path = Path(root_dir) / patent_dir
        
        # Skip if not a directory
        if not patent_path.is_dir():
            continue
            
        # Check if output.txt exists
        txt_file = patent_path / 'txt' / 'output.txt'
        if not txt_file.exists():
            continue
            
        # Create new filename with .md extension
        new_filename = f"{patent_dir}.md"
        output_path = Path(output_dir) / new_filename
        
        # Copy and rename file
        shutil.copy2(txt_file, output_path)

if __name__ == "__main__":
    root_dir = r"D:\python_projects\GPT_Paper\data\OCM_articles_MD_RAW"
    output_dir = r"D:\python_projects\GPT_Paper\data\OCM_articles_MD"
    copy_and_rename_files(root_dir, output_dir)