"""
Standard storage is most expensive
after 30 days or sooner we want to move older uploads to 
NEARLINE storage which is cheaper

gsutil -m -o  "GSUtil:parallel_process_count=5" rewrite -s NEARLINE -O "gs://auroreye-storage-558/seq_2023-03-14T01-09-14/**"
"""

import subprocess
from datetime import datetime, timedelta

BUCKET = "gs://auroreye-storage-558"
MOVE_AFTER_DAYS = 400
STORAGE_CLASS = "NEARLINE"

# list folders
folders = subprocess.check_output(
    ["gsutil", "ls", BUCKET + "/"]
).decode().splitlines()

now = datetime.utcnow()

for folder in folders:
    # folder format: gs://your-bucket/seq_YYYY-MM-DDTHH-MM-SS/
    folder_name = folder.rstrip("/").split("/")[-1]
    if not folder_name.startswith("seq_"):
        continue

    date_str = folder_name[4:]  # remove 'seq_'
    try:
        folder_date = datetime.strptime(date_str, "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        print(f"folder date could not be determined for {folder_name}. SKIPPING")
        continue

    age_days = (now - folder_date).days
    if age_days >= MOVE_AFTER_DAYS:
        print(f"Moving {folder} to {STORAGE_CLASS}")
        subprocess.run([
            "gsutil", "-m",
            "rewrite",
            "-s", STORAGE_CLASS,
            "-O", # bucket level permissions not individual file
            folder + "**"
        ])
