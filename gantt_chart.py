#!/usr/bin/env python3
"""
Timeline scatter chart from catalog_summary.csv.

Reads a CSV with columns: folder, datetime_start, datetime_end, image_count, camera_type, lens_type
Generates a scatter plot with one row per entry, ordered by start date,
coloured by camera type, with date on the x-axis and circular markers
sized by image count (small <500, medium 500-1000, large >1000).
"""

import argparse
import sys
from pathlib import Path

# Force non-interactive Agg backend (avoids _tkinter issues on macOS)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.patches import Patch


# ── date parsing ──────────────────────────────────────────────────────────────

def parse_datetime(val: str):
    """Convert a 'YYYY:MM:DD HH:MM:SS' string to a pandas Timestamp (NaT if invalid)."""
    if pd.isna(val) or str(val).strip().upper() == "N/A":
        return pd.NaT
    try:
        return pd.Timestamp(str(val).strip().replace(":", "-", 2))
    except (ValueError, TypeError):
        return pd.NaT


def is_valid_date(ts) -> bool:
    """Return True if ts is a valid Timestamp that is not the bogus 1980 sentinel."""
    if pd.isna(ts):
        return False
    # The EXIF epoch sentinel (1980-01-01) indicates missing metadata
    if ts.year <= 1980:
        return False
    return True


# ── colour mapping ────────────────────────────────────────────────────────────

# A qualitative palette for camera types
CAMERA_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf",  # teal
]


def build_camera_colour_map(series: pd.Series) -> dict:
    """Assign a colour to each unique camera type."""
    unique = series.dropna().unique()
    unique_sorted = sorted(str(c) for c in unique)
    return {c: CAMERA_PALETTE[i % len(CAMERA_PALETTE)] for i, c in enumerate(unique_sorted)}


# ── marker sizing ─────────────────────────────────────────────────────────────

def marker_size_from_count(count: int) -> float:
    """Return a marker area (points^2) based on image count category."""
    if count < 500:
        return 20.0   # small
    elif count <= 1000:
        return 60.0   # medium
    else:
        return 150.0  # large


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a timeline scatter chart from catalog_summary.csv"
    )
    parser.add_argument(
        "-i", "--input",
        default="catalog_summary.csv",
        help="Path to the input CSV file (default: catalog_summary.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        default="gantt_chart.png",
        help="Path for the output PNG image (default: gantt_chart.png)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Only plot rows from this year (default: 2026). Use 0 to plot all years.",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.is_file():
        print(f"Error: input file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # ── read & clean ──────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    required_cols = {"folder", "datetime_start", "datetime_end", "camera_type", "image_count"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: CSV missing required columns: {', '.join(sorted(missing))}", file=sys.stderr)
        sys.exit(1)

    # Parse dates
    df["start"] = df["datetime_start"].apply(parse_datetime)

    # Filter invalid dates
    before = len(df)
    df = df[df["start"].apply(is_valid_date)].copy()
    skipped = before - len(df)
    if skipped:
        print(f"Skipped {skipped} row(s) with invalid / missing dates.")

    # Filter by year if requested
    if args.year != 0:
        before = len(df)
        df = df[df["start"].dt.year == args.year].copy()
        skipped = before - len(df)
        if skipped:
            print(f"Filtered out {skipped} row(s) not in year {args.year}.")

    if df.empty:
        print("No valid data rows to plot.", file=sys.stderr)
        sys.exit(0)

    # Sort by start date (ascending) – earliest first, which will be at the top
    df.sort_values("start", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Extract short folder name (last path component)
    df["slug"] = df["folder"].apply(lambda p: Path(str(p)).name)

    # ── build colour map ──────────────────────────────────────────────────
    colour_map = build_camera_colour_map(df["camera_type"])
    df["colour"] = df["camera_type"].apply(lambda c: colour_map.get(str(c), "#999999"))

    # ── marker sizes ──────────────────────────────────────────────────────
    df["marker_size"] = df["image_count"].apply(marker_size_from_count)

    # ── plot ──────────────────────────────────────────────────────────────
    n_rows = len(df)
    # Dynamically size the figure: minimum 6" tall, 0.3" per row
    fig_height = max(6, n_rows * 0.30)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Scatter plot: one point per row at the start datetime
    ax.scatter(
        df["start"],
        range(n_rows),
        s=df["marker_size"],
        c=df["colour"],
        edgecolors="white",
        linewidths=0.4,
        alpha=0.85,
        zorder=3,
    )

    # ── axis labels & ticks ───────────────────────────────────────────────
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(df["slug"], fontsize=7)
    ax.invert_yaxis()  # earliest at top

    ax.set_xlabel("Date", fontsize=10)
    ax.set_title("Sequence Timeline", fontsize=13, fontweight="bold")

    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30, ha="right")

    # Grid lines
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    # ── legend: camera colours ────────────────────────────────────────────
    legend_patches = [
        Patch(color=colour, label=cam)
        for cam, colour in sorted(colour_map.items())
    ]
    first_legend = ax.legend(
        handles=legend_patches,
        title="Camera",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(first_legend)

    # ── legend: marker sizes ──────────────────────────────────────────────
    size_legend_elements = [
        plt.scatter([], [], s=marker_size_from_count(250),  c="#555555", edgecolors="white", linewidths=0.4, label="< 500"),
        plt.scatter([], [], s=marker_size_from_count(750),  c="#555555", edgecolors="white", linewidths=0.4, label="500 – 1000"),
        plt.scatter([], [], s=marker_size_from_count(1500), c="#555555", edgecolors="white", linewidths=0.4, label="> 1000"),
    ]
    ax.legend(
        handles=size_legend_elements,
        title="Image count",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.65),
        fontsize=8,
        title_fontsize=9,
    )

    plt.tight_layout()
    out_path = Path(args.output)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Timeline chart saved to: {out_path.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
