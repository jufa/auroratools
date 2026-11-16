import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from exif import Image

def get_exif_metadata(file_path):
    """Extracts date, camera, and lens metadata from a single JPG file."""
    metadata = {
        "file_path": file_path,
        "datetime": None,
        "camera_type": "N/A",
        "lens_type": "N/A"
    }
    try:
        with open(file_path, 'rb') as img_file:
            chunk = img_file.read(256 * 1024) 
            
        img = Image(chunk)
        if img.has_exif:
            metadata["datetime"] = img.get("datetime_original")
            metadata["camera_type"] = img.get("model", "N/A")
            metadata["lens_type"] = img.get("lens_model", "N/A")
    except Exception:
        pass
    return metadata

def print_folder_progress(folder_name, iteration, total, bar_length=40):
    """Prints an inline progress bar tailored to individual folders."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(bar_length * iteration // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # Truncate folder name if it's too long for a standard terminal window
    display_name = folder_name if len(folder_name) <= 30 else f"...{folder_name[-27:]}"
    
    sys.stdout.write(f'\rProcessing: {display_name:<30} |{bar}| {percent}% ({iteration}/{total})')
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Parallelized JPG catalog summary with per-folder progress.")
    parser.add_argument("-d", "--directory", required=True, type=str, help="The parent directory to crawl.")
    parser.add_argument("-o", "--output", required=True, type=str, help="The output path for the summary CSV.")
    args = parser.parse_args()
    
    parent_dir = Path(args.directory).expanduser()
    output_csv = Path(args.output).expanduser()
    
    if not parent_dir.is_dir():
        print(f"Error: The directory '{parent_dir}' does not exist.")
        return

    print(f"Scanning '{parent_dir}' for folders containing JPGs...")
    
    # Find all JPG files first to see what folders we actually need to process
    all_jpgs = list(parent_dir.rglob("*.[jJ][pP][gG]")) + list(parent_dir.rglob("*.[jJ][pP][eE][gG]"))
    
    if not all_jpgs:
        print("No JPG images found.")
        return

    # Map folders to their respective files
    folder_map = {}
    for file_path in all_jpgs:
        folder_map.setdefault(file_path.parent, []).append(file_path)

    print(f"Found {len(all_jpgs)} total images across {len(folder_map)} folders. Beginning extraction...\n")

    summary_data = []

    # Process one folder at a time, but process the files inside that folder in parallel
    for folder, files in folder_map.items():
        total_files = len(files)
        valid_meta = []
        completed_files = 0
        
        # Print initial 0% state for this folder
        print_folder_progress(folder.name, 0, total_files)
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(get_exif_metadata, f): f for f in files}
            
            for future in as_completed(futures):
                meta = future.result()
                if meta["datetime"]:
                    valid_meta.append(meta)
                
                completed_files += 1
                print_folder_progress(folder.name, completed_files, total_files)
        
        # Move down to a new line so the next folder doesn't overwrite this one
        print()

        # Handle summary logic for the folder
        if not valid_meta:
            summary_data.append({
                "folder": str(folder),
                "datetime_start": "N/A",
                "datetime_end": "N/A",
                "image_count": total_files,
                "camera_type": "N/A",
                "lens_type": "N/A"
            })
        else:
            valid_meta.sort(key=lambda x: x["datetime"])
            first_image = valid_meta[0]
            last_image = valid_meta[-1]
            
            summary_data.append({
                "folder": str(folder),
                "datetime_start": first_image["datetime"],
                "datetime_end": last_image["datetime"],
                "image_count": total_files,
                "camera_type": first_image["camera_type"],
                "lens_type": first_image["lens_type"]
            })

    # Write to CSV
    headers = ["folder", "datetime_start", "datetime_end", "image_count", "camera_type", "lens_type"]
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(summary_data)
            
        print(f"\nSuccess! Catalog summary saved to: {output_csv.resolve()}")
    except Exception as e:
        print(f"\nError writing to CSV file: {e}")

if __name__ == "__main__":
    main()