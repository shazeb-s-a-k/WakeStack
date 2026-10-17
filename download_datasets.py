"""
=============================================================================
                  openWakeWord Training Datasets Downloader
=============================================================================
Downloads the required precomputed negative feature arrays (ACAV100M 2,000 hrs)
and validation dataset from Hugging Face with progress tracking.

Files:
  1. openwakeword_features_ACAV100M_2000_hrs_16bit.npy (~16.1 GB)
  2. validation_set_features.npy (~176 MB)
=============================================================================
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS = {
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": {
        "url": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        "expected_bytes": 17280000128,
        "description": "ACAV100M 2,000 Hours Negative Features (17.28 GB)"
    },
    "validation_set_features.npy": {
        "url": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
        "expected_bytes": 184836608,
        "description": "False Positive Validation Features (184 MB)"
    }
}

class DownloadProgressBar:
    def __init__(self, filename, total_size):
        self.filename = filename
        self.total_size = total_size
        self.start_time = time.time()
        self.last_update = 0

    def __call__(self, block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size <= 0:
            total_size = self.total_size

        now = time.time()
        if now - self.last_update < 0.5 and downloaded < total_size:
            return
        self.last_update = now

        percent = min(100.0, downloaded * 100.0 / total_size) if total_size > 0 else 0
        elapsed = now - self.start_time
        speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0

        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)

        if total_mb >= 1024:
            size_str = f"{downloaded_mb/1024:.2f} / {total_mb/1024:.2f} GB"
        else:
            size_str = f"{downloaded_mb:.1f} / {total_mb:.1f} MB"

        # 30-character progress bar
        bar_len = 30
        filled = int(bar_len * percent / 100)
        bar = "=" * filled + "-" * (bar_len - filled)

        sys.stdout.write(f"\r[{bar}] {percent:5.1f}% | {size_str} | {speed:.1f} MB/s ")
        sys.stdout.flush()

def download_dataset(filename, info):
    filepath = PROJECT_ROOT / filename
    print(f"\nTarget: {info['description']}")
    print(f"Path  : {filepath}")

    if filepath.exists():
        size = filepath.stat().st_size
        if size >= info["expected_bytes"] * 0.99:
            print(f"[OK] File already exists and matches expected size ({size / (1024**2):.1f} MB). Skipping.")
            return

    print(f"Downloading from Hugging Face...")
    print(f"URL: {info['url']}")
    
    progress = DownloadProgressBar(filename, info["expected_bytes"])
    try:
        urllib.request.urlretrieve(info["url"], str(filepath), reporthook=progress)
        print(f"\n[OK] Successfully downloaded {filename}!")
    except Exception as e:
        print(f"\n[ERROR] Failed to download {filename}: {e}")
        if filepath.exists():
            filepath.unlink()
        sys.exit(1)

def main():
    print("=" * 70)
    print("OPENWAKEWORD DATASET DOWNLOADER".center(70))
    print("=" * 70)

    for fname, info in DATASETS.items():
        download_dataset(fname, info)

    print("\n" + "=" * 70)
    print("All training datasets verified and ready!".center(70))
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
