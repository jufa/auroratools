from datetime import datetime
import statistics
import re

LOG_FILE = "sequence_log.txt"
JITTER_TOLERANCE = 1.0  # seconds

timestamp_re = re.compile(r"\[(.*?)\]")

timestamps = []
lines = []

with open(LOG_FILE, "r") as f:
    for line in f:
        match = timestamp_re.search(line)
        if match:
            ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            timestamps.append(ts)
            lines.append(line.rstrip())

# Compute deltas (seconds) between consecutive timestamps
deltas = [
    (timestamps[i] - timestamps[i - 1]).total_seconds()
    for i in range(1, len(timestamps))
]

if not deltas:
    print("Not enough data to compute differences.")
    exit(0)

# Use median as the "typical" interval (robust against outliers)
typical_delta = statistics.median(deltas)

print(f"Typical interval: {typical_delta:.2f} s\n")
print("Anomalies:")

found = False
skip_tally = [0] * 100
for i, delta in enumerate(deltas, start=1):
    interval_time = int(abs(delta - typical_delta))
    print(f"{delta:3.0f},{lines[i].split("/")[1].strip()}\t"+"#"*int(delta))
    if interval_time > JITTER_TOLERANCE:
        found = True
        if(interval_time > len(skip_tally)):
            interval_time = len(skip_tally) - 1
        skip_tally[interval_time] = skip_tally[interval_time] + 1
        # print(
        #     f"Line {i+1}: Δt = {delta:.2f} s "
        #     f"(expected ~{typical_delta:.2f} s)\n"
        #     f"  {lines[i]}"
        # )
if not found:
    print("None found.")
print(f"\nTotal anomalies: {sum(skip_tally)}\n")
for i, count in enumerate(skip_tally):
    if count > 0:
        print(f"{i:3.0f}s   " + "#" * int(count))
