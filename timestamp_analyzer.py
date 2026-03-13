#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime
import sys
from collections import defaultdict

def get_file_birthtime(filepath: Path) -> float:
    """Get the most accurate creation timestamp for a file."""
    stat = filepath.stat()
    # On macOS, st_birthtime is available; on Linux, fall back to st_ctime
    if hasattr(stat, 'st_birthtime'):
        return stat.st_birthtime
    else:
        return stat.st_ctime

def analyze_timestamps(folder: Path, file_extension: str = "tif"):
    """Analyze timestamp differences between sequential image files."""
    # Find all files with specified extension
    pattern = f"*.{file_extension}"
    files = sorted(folder.glob(pattern))
    
    if len(files) < 2:
        print(f"Error: Found only {len(files)} {file_extension} file(s). Need at least 2 files.")
        sys.exit(1)
    
    print(f"Found {len(files)} {file_extension} files in {folder}")
    
    # Extract timestamps
    timestamps = []
    for f in files:
        try:
            ts = get_file_birthtime(f)
            timestamps.append(ts)
        except Exception as e:
            print(f"Warning: Could not read timestamp for {f.name}: {e}")
    
    if len(timestamps) < 2:
        print("Error: Not enough valid timestamps found.")
        sys.exit(1)
    
    # Calculate differences in milliseconds
    differences = []
    for i in range(1, len(timestamps)):
        diff_ms = (timestamps[i] - timestamps[i-1]) * 1000  # convert to ms
        differences.append(diff_ms)
    
    # Statistics
    min_diff = min(differences)
    max_diff = max(differences)
    avg_diff = sum(differences) / len(differences)
    
    print(f"\nStatistics:")
    print(f"  Total intervals: {len(differences)}")
    print(f"  Min difference: {min_diff:.2f} ms")
    print(f"  Max difference: {max_diff:.2f} ms")
    print(f"  Avg difference: {avg_diff:.2f} ms")
    
    # Create histogram with 20 bins of 100ms each (0-2000ms)
    bin_size = 100  # ms
    num_bins = 20
    max_range = bin_size * num_bins  # 2000ms
    
    bins = [0] * num_bins
    underflow = []  # < 0ms (shouldn't happen but just in case)
    overflow = []   # >= 2000ms
    
    for diff in differences:
        if diff < 0:
            underflow.append(diff)
        elif diff >= max_range:
            overflow.append(diff)
        else:
            bin_idx = min(int(diff / bin_size), num_bins - 1)
            bins[bin_idx] += 1
    
    # Display ASCII histogram
    print(f"\nHistogram (bin size: {bin_size}ms, range: 0-{max_range}ms)")
    print("=" * 70)
    
    max_count = max(bins) if max(bins) > 0 else 1
    bar_width = 50
    
    for i, count in enumerate(bins):
        range_start = i * bin_size
        range_end = (i + 1) * bin_size
        bar_length = int((count / max_count) * bar_width) if count > 0 else 0
        bar = '█' * bar_length
        percentage = (count / len(differences)) * 100
        print(f"{range_start:4d}-{range_end:4d}ms | {bar:<{bar_width}} | {count:5d} ({percentage:5.2f}%)")
    
    # Report extreme outliers
    if underflow:
        print(f"\n⚠ Extreme outliers (UNDERFLOW, < 0ms): {len(underflow)} cases")
        for diff in sorted(underflow)[:10]:  # show first 10
            print(f"  {diff:.2f} ms")
        if len(underflow) > 10:
            print(f"  ... and {len(underflow) - 10} more")
    
    if overflow:
        print(f"\n⚠ Extreme outliers (OVERFLOW, >= {max_range}ms): {len(overflow)} cases")
        for diff in sorted(overflow)[:10]:  # show first 10
            print(f"  {diff:.2f} ms")
        if len(overflow) > 10:
            print(f"  ... and {len(overflow) - 10} more")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze timestamp differences between sequential image files"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to folder containing image files"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="tif",
        help="File extension to analyze (default: tif)"
    )
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    analyze_timestamps(folder_path, args.type)

if __name__ == "__main__":
    main()
