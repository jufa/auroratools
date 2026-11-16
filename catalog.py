import argparse
import csv
from pathlib import Path
from exif import Image

def get_exif_metadata(file_path):
    """Extracts specific EXIF metadata from a JPG file."""
    metadata = {
        "File Path": str(file_path),
        "Created Date": "N/A",
        "ISO": "N/A",
        "Exposure Time": "N/A",
        "Camera Type": "N/A",
        "Lens Type": "N/A"
    }
    
    try:
        with open(file_path, 'rb') as img_file:
            img = Image(img_file)
            
        if img.has_exif:
            # Mapping desired metadata to EXIF tags
            metadata["Created Date"] = img.get("datetime_original", "N/A")
            metadata["ISO"] = img.get("iso", "N/A")
            
            # Exposure time is often stored as a float/fraction; format nicely if present
            exposure = img.get("exposure_time")
            if exposure:
                metadata["Exposure Time"] = f"{exposure}s" if exposure < 1 else str(exposure)
                
            metadata["Camera Type"] = img.get("model", "N/A")
            metadata["Lens Type"] = img.get("lens_model", "N/A")
            
    except Exception as e:
        # If a file is corrupt or unreadable, we log it and skip to the next
        print(f"Warning: Could not read metadata for {file_path}. Error: {e}")
        
    return metadata

def main():
    # Setup CLI Arguments
    parser = argparse.ArgumentParser(description="Extract EXIF metadata from JPGs in a directory recursively.")
    parser.add_argument("-d", "--directory", required=True, type=str, help="The target directory to crawl for JPGs.")
    parser.add_argument("-o", "--output", required=True, type=str, help="The output path and filename for the CSV.")
    
    args = parser.parse_args()
    
    target_dir = Path(args.directory)
    output_csv = Path(args.output)
    
    if not target_dir.is_dir():
        print(f"Error: The directory '{target_dir}' does not exist.")
        return

    print(f"Scanning '{target_dir}' for JPG images...")
    
    # Define CSV column headers
    headers = ["File Path", "Created Date", "ISO", "Exposure Time", "Camera Type", "Lens Type"]
    
    # Find all .jpg and .jpeg files (case-insensitive) recursively
    jpg_files = list(target_dir.rglob("*.[jJ][pP][gG]")) + list(target_dir.rglob("*.[jJ][pP][eE][gG]"))
    
    if not jpg_files:
        print("No JPG images found in the specified directory.")
        return
        
    print(f"Found {len(jpg_files)} images. Extracting metadata...")

    # Write data to CSV
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            
            for file_path in jpg_files:
                file_meta = get_exif_metadata(file_path)
                writer.writerow(file_meta)
                
        print(f"Success! Metadata successfully saved to: {output_csv.resolve()}")
        
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

if __name__ == "__main__":
    main()