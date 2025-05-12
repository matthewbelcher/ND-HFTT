#!/usr/bin/env python3
import subprocess
import glob
import os
import sys

# --- CONFIGURATION ---
# Point this at the directory containing your .gz files:
INPUT_DIR = os.path.abspath("../../HFTT_Data")
# Adjust the Python executable or script name/path if needed:
PYTHON      = sys.executable      # e.g. "/usr/bin/python3"
CLEAN_SCRIPT = "clean_data.py"
# ----------------------

def process_file(gz_path):
    """
    1) gunzip the file in place,
    2) run clean_data.py on it,
    3) gzip it again.
    """
    raw_path = gz_path[:-3]  # strip ".gz"
    print(f">>> Processing {os.path.basename(gz_path)}")


    # 1) Decompress
    print(f"    • Unzipping → {raw_path}")
    subprocess.run(["gunzip", "-f", gz_path], check=True)

    # 2) Filter
    print(f"    • Cleaning → {raw_path}")
    subprocess.run([PYTHON, CLEAN_SCRIPT, raw_path], check=True)

    # 3) Re-compress
    print(f"    • Gzipping  → {raw_path}.gz")
    subprocess.run(["gzip", "-f", raw_path], check=True)

    print(f"    ✓ Done with {os.path.basename(gz_path)}\n")

def main():
    pattern = os.path.join(INPUT_DIR, "**", "*.gz")
    gz_files = glob.glob(pattern, recursive=True)
    if not gz_files:
        print(f"No .gz files found under {INPUT_DIR}")
        return

    for gz in sorted(gz_files):
        process_file(gz)

if __name__ == "__main__":
    main()
