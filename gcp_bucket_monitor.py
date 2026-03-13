"""
GCP Bucket Monitor
==================
Runs as a background daemon on macOS. Periodically checks the GCP bucket for:

  1. NEW folders that have appeared since the last check
  2. STALE folders — folders that exist but have had no new files written
     to them in the last STALE_MINUTES minutes (i.e. an upload may have
     stalled or been abandoned)

Sends an email notification to the configured recipient in each case.

Usage:
    python gcp_bucket_monitor.py [--interval SECONDS] [--stale-minutes MINUTES]

Run in the background (persists after terminal close):
    nohup python gcp_bucket_monitor.py >> ~/logs/gcp_bucket_monitor.log 2>&1 &

Or install as a launchd service — see the accompanying
com.auroratools.gcpbucketmonitor.plist file.


# Quick status check (default 20 rows)
python gcp_bucket_monitor.py

# Custom row count
python gcp_bucket_monitor.py --status 10

# Start the background daemon
python gcp_bucket_monitor.py --daemon --interval 300 --stale-minutes 30

# Install as a persistent launchd service (survives reboots)
mkdir -p ~/logs
cp com.auroratools.gcpbucketmonitor.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.auroratools.gcpbucketmonitor.plist


"""

import subprocess
import time
import smtplib
import logging
import argparse
import json
import os
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these or override via environment variables
# ---------------------------------------------------------------------------

BUCKET = os.environ.get("GCP_BUCKET", "gs://auroreye-storage-558")

# How often to poll (seconds). Override with --interval or $MONITOR_INTERVAL.
DEFAULT_CHECK_INTERVAL_SECONDS = 0  # in seconds, 0 is once only

# A folder is considered "stale" if its most-recent object is older than this.
DEFAULT_STALE_MINUTES = 10

# Email settings — populate via environment variables to keep secrets out of
# source control.  SMTP_PASSWORD should be an app-specific password.
EMAIL_FROM    = os.environ.get("SMTP_FROM",     "jeremy@jufaintermedia.com")
EMAIL_TO      = os.environ.get("ALERT_EMAIL",   "jeremy@jufaintermedia.com")
SMTP_HOST     = os.environ.get("SMTP_HOST",     "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER     = os.environ.get("SMTP_USER",     EMAIL_FROM)
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")  # set in environment!

# Path to persist state between runs so we don't re-alert on restart.
STATE_FILE = Path(os.environ.get("MONITOR_STATE_FILE",
                                  Path.home() / ".gcp_bucket_monitor_state.json"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load persisted state from disk."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception as exc:
            log.warning("Could not read state file: %s", exc)
    return {
        "known_folders": [],        # folders seen on previous runs
        "stale_alerted": [],        # folders we already sent a stale alert for
        "new_folder_alerted": [],   # folders we already sent a new-folder alert for
        "sequence_log_counts": {},  # folder -> line count of sequence_log.txt
    }


def save_state(state: dict) -> None:
    """Persist state to disk."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as exc:
        log.warning("Could not write state file: %s", exc)


# ---------------------------------------------------------------------------
# GCP helpers
# ---------------------------------------------------------------------------

def list_folders(bucket: str) -> list[str]:
    """
    Return a list of top-level folder names sorted newest-first by name.
    No-GPS folders (seq_0000-*) are sorted to the top since they can't be
    sorted by date — they will be distinguished by the NEW marker instead.
    """
    try:
        output = subprocess.check_output(
            ["gsutil", "ls", bucket + "/"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError as exc:
        log.error("gsutil ls failed: %s", exc)
        return []

    folders = []
    for line in output.splitlines():
        name = line.rstrip("/").split("/")[-1]
        if name:
            folders.append(name)

    # Sort: no-GPS (zero-date) folders first, then real timestamps newest-first
    def sort_key(name: str) -> tuple:
        return (0 if is_no_gps_folder(name) else 1, name)

    return sorted(folders, key=sort_key, reverse=True)


def parse_folder_datetime(folder_name: str) -> datetime | None:
    """
    Parse the embedded timestamp from a folder named seq_YYYY-MM-DDTHH-MM-SS.
    Returns None for no-GPS folders (seq_0000-00-00T00-00-00_*) and other
    unparseable names.
    """
    if not folder_name.startswith("seq_"):
        return None
    try:
        # Strip any trailing suffix after the timestamp (e.g. seq_0000-00-00T00-00-00_extra)
        date_part = folder_name[4:].split("_")[0]
        dt = datetime.strptime(date_part, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
        # Reject the zero-date placeholder used when GPS has no fix
        if dt.year == 0 or date_part.startswith("0000-"):
            return None
        return dt
    except ValueError:
        return None


def is_no_gps_folder(folder_name: str) -> bool:
    """Return True for folders with the zero-date GPS placeholder prefix."""
    return folder_name.startswith("seq_0000-")


def fetch_sequence_log_count(bucket: str, folder_name: str) -> int | None:
    """
    Fetch sequence_log.txt from the bucket and return its line count.
    Returns None if the file doesn't exist or can't be read.
    """
    uri = f"{bucket}/{folder_name}/sequence_log.txt"
    try:
        output = subprocess.check_output(
            ["gsutil", "cat", uri],
            stderr=subprocess.DEVNULL,
        ).decode()
        return sum(1 for line in output.splitlines() if line.strip())
    except subprocess.CalledProcessError:
        return None


def get_sequence_log_counts(
    bucket: str,
    folders: list[str],
    state: dict,
) -> dict[str, int | None]:
    """
    Return a dict of folder -> line count for sequence_log.txt.
    Uses cached values from state for folders already seen;
    only fetches from GCS for new folders.
    """
    cache: dict[str, int | None] = state.setdefault("sequence_log_counts", {})

    for folder in folders:
        if folder in cache:
            continue  # already cached, skip the network call
        log.info("Fetching sequence_log.txt for new folder: %s", folder)
        cache[folder] = fetch_sequence_log_count(bucket, folder)

    return cache


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def send_email(subject: str, body: str) -> None:
    return
    """Send a plain-text alert email."""
    if not SMTP_PASSWORD:
        log.warning("SMTP_PASSWORD not set — email not sent. Subject: %s", subject)
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        log.info("Email sent: %s", subject)
    except Exception as exc:
        log.error("Failed to send email: %s", exc)


# ---------------------------------------------------------------------------
# Check logic
# ---------------------------------------------------------------------------

def check_new_folders(current_folders: list[str], state: dict) -> None:
    """Alert on any folder not seen in a previous run."""
    known = set(state["known_folders"])
    already_alerted = set(state["new_folder_alerted"])

    for folder in current_folders:
        if folder not in known and folder not in already_alerted:
            log.info("NEW folder detected: %s", folder)
            send_email(
                subject=f"[AurOrEye] New GCP folder: {folder}",
                body=(
                    f"A new folder has been created in the GCP bucket.\n\n"
                    f"  Bucket : {BUCKET}\n"
                    f"  Folder : {folder}\n"
                    f"  Detected at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} local\n"
                ),
            )
            state["new_folder_alerted"].append(folder)

    # Update known folders list
    state["known_folders"] = sorted(set(state["known_folders"]) | set(current_folders))


def check_stale_folders(
    current_folders: list[str],
    state: dict,
    stale_minutes: int,
) -> None:
    """
    Alert on folders whose name-encoded creation time is older than stale_minutes
    AND that are newly seen (i.e. a sequence started but apparently stopped uploading).
    Only alerts once per folder.
    """
    already_alerted = set(state["stale_alerted"])
    known = set(state["known_folders"])
    now = datetime.now(tz=timezone.utc)

    for folder in current_folders:
        if folder in already_alerted:
            continue
        # Only check folders we already knew about from a previous cycle —
        # a brand-new folder gets a free pass until the next poll.
        if folder not in known:
            continue

        created = parse_folder_datetime(folder)
        if created is None:
            continue

        age_minutes = (now - created).total_seconds() / 60
        if age_minutes >= stale_minutes:
            log.info("STALE folder: %s (created %.1f min ago)", folder, age_minutes)
            send_email(
                subject=f"[AurOrEye] Stale GCP folder: {folder}",
                body=(
                    f"A sequence folder was created {age_minutes:.0f} minutes ago "
                    f"and may have stalled.\n\n"
                    f"  Bucket    : {BUCKET}\n"
                    f"  Folder    : {folder}\n"
                    f"  Created   : {created.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                    f"  Threshold : {stale_minutes} minutes\n"
                    f"  Checked at: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                ),
            )
            state["stale_alerted"].append(folder)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(check_interval: int, stale_minutes: int) -> None:
    log.info(
        "GCP Bucket Monitor started. bucket=%s  interval=%ds  stale=%dmin",
        BUCKET, check_interval, stale_minutes,
    )

    while True:
        log.info("--- Checking bucket ---")
        state = load_state()

        current_folders = list_folders(BUCKET)
        log.info("Found %d top-level folders", len(current_folders))

        check_new_folders(current_folders, state)
        check_stale_folders(current_folders, state, stale_minutes)

        save_state(state)
        log.info("Next check in %d seconds", check_interval)
        if check_interval <= 0:
            log.info("Check interval is non-positive, exiting.")
            break
        if check_interval > 0 and check_interval < 60:
            log.warning(f"Check interval is very short ({check_interval} seconds) — defaulting to 60 seconds.")
            check_interval = 60
        time.sleep(check_interval)


# ---------------------------------------------------------------------------
# One-shot status report
# ---------------------------------------------------------------------------

def print_status(rows: int) -> None:
    """
    Print a table of the N most-recently-created folders, newest first.
    Uses only the folder name as a date proxy — single cheap gsutil ls call.
    Persists state so subsequent runs can highlight NEW folders.
    """
    state = load_state()
    known = set(state["known_folders"])

    all_folders = list_folders(BUCKET)  # already sorted newest-first
    if not all_folders:
        print("No folders found.")
        return

    # Parse dates and take the top N
    results: list[tuple[str, datetime | None, bool]] = []
    for folder in all_folders:
        dt = parse_folder_datetime(folder)
        is_new = folder not in known
        results.append((folder, dt, is_new))
        if len(results) == rows:
            break

    # Fetch sequence_log.txt line counts — cached for known folders, live for new ones
    log_counts = get_sequence_log_counts(BUCKET, [r[0] for r in results], state)

    # Update state with everything we saw (not just the top N)
    state["known_folders"] = sorted(set(state["known_folders"]) | set(all_folders))
    save_state(state)

    now = datetime.now(tz=timezone.utc)
    col_folder = max(len(r[0]) for r in results)
    col_folder = max(col_folder, len("Folder"))

    header = f"{'Folder':<{col_folder}}  {'Created (UTC)':<22}  {'Age':<16}  {'Files':>6}  New?"
    print()
    print(header)
    print("-" * (len(header) + 4))
    for folder, created, is_new in results:
        if created is None:
            ts_str = "no GPS fix" if is_no_gps_folder(folder) else "unknown"
            age_str = "—"
        else:
            ts_str = created.strftime("%Y-%m-%d %H:%M:%S")
            delta = now - created
            total_minutes = int(delta.total_seconds() // 60)
            if total_minutes < 60:
                age_str = f"{total_minutes}m ago"
            elif total_minutes < 1440:
                age_str = f"{total_minutes // 60}h {total_minutes % 60}m ago"
            else:
                age_str = f"{total_minutes // 1440}d {(total_minutes % 1440) // 60}h ago"
        count = log_counts.get(folder)
        count_str = str(count) if count is not None else "—"
        new_marker = "  *** NEW" if is_new else ""
        print(f"{folder:<{col_folder}}  {ts_str:<22}  {age_str:<16}  {count_str:>6}{new_marker}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor a GCP bucket for new/stale folders, or print a status report.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_CHECK_INTERVAL_SECONDS,
        help=f"How often to poll, in seconds (default: {DEFAULT_CHECK_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--stale-minutes",
        type=int,
        default=DEFAULT_STALE_MINUTES,
        dest="stale_minutes",
        help=f"Alert if no new files written for this many minutes (default: {DEFAULT_STALE_MINUTES})",
    )
    parser.add_argument(
        "--status",
        nargs="?",
        const=20,
        type=int,
        metavar="N",
        help="Print a one-shot status table of the N most-recently-updated folders (default: 20) and exit.",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a background monitor loop instead of printing the status table.",
    )
    args = parser.parse_args()

    if args.status is not None or (args.status is None and not args.daemon):
        print_status(rows=args.status if args.status is not None else 20)
    else:
        run(check_interval=args.interval, stale_minutes=args.stale_minutes)
