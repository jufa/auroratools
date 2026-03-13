"""
sequence_summary.py
====================
Scans all seq_* folders under a root directory and builds (or updates) a
summary CSV:  sequence_summary.csv

Columns written:
    folder, image_count, start_time, end_time, exposure_time,
    photographic_sensitivity, latitude, latitude_ref,
    longitude, longitude_ref, camera_model

Idempotent: folders already present in the summary CSV are skipped unless
--refresh is passed.

Usage:
    python sequence_summary.py
    python sequence_summary.py --root '/Volumes/T7 Shield/AurorEye'
    python sequence_summary.py --refresh   # re-scan all folders
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ROOT = "/Volumes/T7 Shield/AurorEye"
SUMMARY_FILENAME = "sequence_summary.csv"

SUMMARY_COLUMNS = [
    "folder",
    "image_count",
    "start_time",
    "end_time",
    "exposure_time",
    "photographic_sensitivity",
    "latitude",
    "latitude_ref",
    "longitude",
    "longitude_ref",
    "camera_model",
]

# Per-camera column mappings  (model substring, lowercase) -> field names
CAMERA_COLUMN_MAPS = {
    "m100": {
        "datetime":                 "datetime_original",  # already a combined datetime, UTC
        "exposure_time":            "exposure_time",
        "photographic_sensitivity": "photographic_sensitivity",
        "model":                    "model",
        "latitude":                 "gps_latitude",
        "latitude_ref":             "gps_latitude_ref",
        "longitude":                "gps_longitude",
        "longitude_ref":            "gps_longitude_ref",
    },
    "zv-e10": {
        "date":                     "date",               # combined with "time" below
        "time":                     "time",
        "exposure_time":            "exposure_time",
        "photographic_sensitivity": "photographic_sensitivity",
        "model":                    "model",
        "latitude":                 "latitude",
        "latitude_ref":             "latitude_ref",
        "longitude":                "longitude",
        "longitude_ref":            "longitude_ref",
    },
}

# Fallback probe order if model is unrecognised
FALLBACK_GPS_CANDIDATES = [
    ("latitude",     "latitude_ref",     "longitude",     "longitude_ref"),
    ("gps_latitude", "gps_latitude_ref", "gps_longitude", "gps_longitude_ref"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def column_map_for_model(model: str) -> dict | None:
    """Return the column-name map for a given camera model string, or None."""
    m = model.lower()
    # Treat all Canon M-series (m100, m200, m50, etc.) identically
    if re.search(r'\bm\d+\b', m):
        m = "m100"
    for key, cmap in CAMERA_COLUMN_MAPS.items():
        if key in m:
            return cmap
    return None


def extract_folder_data(folder_path: Path) -> dict:
    """
    Read metadata.csv from folder_path and return a summary dict.
    All data fields are empty strings if metadata.csv is missing or unreadable.
    """
    empty = {col: "" for col in SUMMARY_COLUMNS}
    empty["folder"] = folder_path.name

    meta_path = folder_path / "metadata.csv"
    if not meta_path.exists():
        return empty

    try:
        with open(meta_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            # Strip BOM / stray whitespace from header names
            if reader.fieldnames:
                reader.fieldnames = [h.strip() for h in reader.fieldnames]
            rows = list(reader)
    except Exception as exc:
        print(f"  WARNING: could not read {meta_path}: {exc}", file=sys.stderr)
        return empty

    if not rows:
        return empty

    first = rows[0]
    last  = rows[-1]

    # Determine camera model from first data row
    model_raw = first.get("model", "").strip()
    cmap = column_map_for_model(model_raw)

    if cmap:
        exp_col    = cmap["exposure_time"]
        iso_col    = cmap["photographic_sensitivity"]
        lat_col    = cmap["latitude"]
        latref_col = cmap["latitude_ref"]
        lon_col    = cmap["longitude"]
        lonref_col = cmap["longitude_ref"]
    else:
        # Unknown model — probe which GPS columns are actually present
        exp_col  = "exposure_time"
        iso_col  = "photographic_sensitivity"
        lat_col = latref_col = lon_col = lonref_col = ""
        for lat, latref, lon, lonref in FALLBACK_GPS_CANDIDATES:
            if lat in first:
                lat_col, latref_col, lon_col, lonref_col = lat, latref, lon, lonref
                break
        if not lat_col:
            print(f"  WARNING: unrecognised model '{model_raw}' and no GPS columns found in {folder_path.name}",
                  file=sys.stderr)

    def get(row: dict, col: str) -> str:
        return row.get(col, "").strip() if col else ""

    def make_datetime(row: dict) -> str:
        """
        Return an ISO-8601 UTC datetime string for a row, depending on camera format.
        M100:   datetime_original column already contains a combined value.
        ZV-E10: concatenate date + time columns.
        Unknown: fall back to whatever we have.
        """
        cmap_key = None
        m = model_raw.lower()
        for key in CAMERA_COLUMN_MAPS:
            if key in m:
                cmap_key = key
                break

        if cmap_key == "m100":
            raw = get(row, "datetime_original")
            # Format is "1980:01:01 00:02:49" — normalise to ISO with Z suffix
            raw = raw.replace(":", "-", 2) if raw.count(":") >= 4 else raw
            return (raw.replace(" ", "T") + "Z") if raw else ""

        if cmap_key == "zv-e10":
            date_part = get(row, "date")   # e.g. "2025-12-06"
            time_part = get(row, "time")   # e.g. "07:28:19"
            if date_part and time_part:
                return f"{date_part}T{time_part}Z"
            return date_part or time_part

        # Fallback: try datetime_original, then date+time, then time alone
        if "datetime_original" in row:
            raw = get(row, "datetime_original")
            raw = raw.replace(":", "-", 2) if raw.count(":") >= 4 else raw
            return (raw.replace(" ", "T") + "Z") if raw else ""
        date_part = get(row, "date")
        time_part = get(row, "time")
        if date_part and time_part:
            return f"{date_part}T{time_part}Z"
        return time_part or date_part

    return {
        "folder":                   folder_path.name,
        "image_count":              str(len(rows)),
        "start_time":               make_datetime(first),
        "end_time":                 make_datetime(last),
        "exposure_time":            get(first, exp_col),
        "photographic_sensitivity": get(first, iso_col),
        "latitude":                 get(first, lat_col),
        "latitude_ref":             get(first, latref_col),
        "longitude":                get(first, lon_col),
        "longitude_ref":            get(first, lonref_col),
        "camera_model":             model_raw,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build/update a summary CSV of all seq_* folders.")
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=f"Root folder containing seq_* directories (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-scan all folders, ignoring any existing summary entries.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root path not found: {root}", file=sys.stderr)
        sys.exit(1)

    summary_path = root / SUMMARY_FILENAME

    # Load existing summary to skip already-processed folders
    existing: dict[str, dict] = {}
    if summary_path.exists() and not args.refresh:
        with open(summary_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["folder"]] = row
        print(f"Loaded {len(existing)} existing entries from {summary_path.name}")

    # Discover seq_* folders, newest-first
    seq_folders = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("seq_")],
        reverse=True,
    )
    print(f"Found {len(seq_folders)} seq_* folders under {root}")

    results: dict[str, dict] = dict(existing)
    new_count = 0

    for folder in seq_folders:
        if folder.name in existing:
            print(f"  SKIP  {folder.name}")
            continue
        print(f"  SCAN  {folder.name}")
        results[folder.name] = extract_folder_data(folder)
        new_count += 1

    # Write summary — newest folder first
    sorted_rows = sorted(results.values(), key=lambda r: r["folder"], reverse=True)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"\nDone — {new_count} new folder(s) added. Summary: {summary_path}")


if __name__ == "__main__":
    main()
