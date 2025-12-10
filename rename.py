import os
import re
import argparse

def rename_files(folder, start_num, end_num, dry_run):
    # Get a list of files in the folder
    files = sorted(os.listdir(folder))
    
    # Filter files that have a numerical sequence in their names
    numbered_files = [f for f in files if re.search(r'\d+', f)]
    if not numbered_files:
        print("No files with numbers found in the specified folder.")
        return
    
    # Default start and end numbers based on available files
    if start_num is None or end_num is None:
        start_num = 0 if start_num is None else start_num
        end_num = len(numbered_files) if end_num is None else end_num
    
    # Ensure start_num and end_num are within range
    numbered_files = numbered_files[start_num:end_num]
    
    # Generate new file names
    for idx, old_name in enumerate(numbered_files):
        new_name = f"{idx:08d}.jpg"  # Zero-padded 8-digit filename
        old_path = os.path.join(folder, old_name)
        new_path = os.path.join(folder, new_name)
        
        if dry_run:
            print(f"[DRY RUN] Would rename: {old_path} -> {new_path}")
        else:
            print(f"Renaming: {old_path} -> {new_path}")
            os.rename(old_path, new_path)

def main():
    parser = argparse.ArgumentParser(description="Sequentially rename files with numerical filenames.")
    parser.add_argument(
        "-f", "--folder", type=str, default=".", 
        help="Folder containing the files to rename (default: current folder)."
    )
    parser.add_argument(
        "-s", "--start", type=int, default=None, 
        help="Start index of the files to rename (default: auto-detect first)."
    )
    parser.add_argument(
        "-e", "--end", type=int, default=None, 
        help="End index of the files to rename (default: auto-detect last)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", 
        help="Perform a dry run without renaming files."
    )
    args = parser.parse_args()
    
    rename_files(args.folder, args.start, args.end, args.dry_run)

if __name__ == "__main__":
    main()
