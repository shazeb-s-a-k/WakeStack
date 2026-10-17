"""
=============================================================================
                  openWakeWord Cache Cleaner Utility
=============================================================================
Cleans up intermediate generated synthetic audio clips (.wav) and temporary
feature cache files (.npy) generated during the training process to reclaim
disk space (typically 2-5 GB per training run).

SAFE BY DESIGN:
- NEVER touches final exported models (.onnx / .tflite).
- NEVER touches ACAV100M features or validation datasets.
- NEVER touches Piper models or impulse filters.

Usage:
    python clean_cache.py                  # Cleans cache for the active MODEL_NAME in config.py
    python clean_cache.py --all            # Cleans all intermediate model caches in models/
    python clean_cache.py --model <name>   # Cleans intermediate cache for a specific model
=============================================================================
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

def get_dir_size_and_count(dir_path):
    total_size = 0
    count = 0
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_size += os.path.getsize(fp)
                count += 1
            except OSError:
                pass
    return count, total_size

def clean_model_cache(model_name):
    target_dir = Path(config.MODELS_DIR) / model_name
    
    if not target_dir.exists():
        print(f"[INFO] No intermediate cache directory found for '{model_name}' at: {target_dir}")
        return 0, 0

    print(f"Scanning intermediate files in: {target_dir}...")
    file_count, byte_size = get_dir_size_and_count(target_dir)
    
    if file_count == 0:
        print(f"[INFO] Cache directory is already empty: {target_dir}")
        try:
            target_dir.rmdir()
        except OSError:
            pass
        return 0, 0

    size_mb = byte_size / (1024**2)
    print(f"Found {file_count:,} temporary files ({size_mb:.2f} MB). Deleting...")
    
    try:
        shutil.rmtree(target_dir)
        print(f"[SUCCESS] Cleaned {file_count:,} cached files for '{model_name}'. Reclaimed {size_mb:.2f} MB!")
        return file_count, byte_size
    except Exception as e:
        print(f"[ERROR] Failed to remove {target_dir}: {e}")
        return 0, 0

def clean_temp_yaml_files():
    """Cleans up old generated_*.yml files in project root."""
    cleaned = 0
    for yml in PROJECT_ROOT.glob("generated_*.yml"):
        try:
            yml.unlink()
            cleaned += 1
        except OSError:
            pass
    if cleaned > 0:
        print(f"[INFO] Cleaned {cleaned} old generated YAML files.")

def main():
    parser = argparse.ArgumentParser(description="Clean intermediate openWakeWord audio clips and cache.")
    parser.add_argument("--model", type=str, default=None, help="Model name whose cache to clean")
    parser.add_argument("--all", action="store_true", help="Clean all model subdirectories in models/")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("OPENWAKEWORD CACHE CLEANER".center(70))
    print("="*70)

    total_files = 0
    total_bytes = 0

    if args.all:
        models_dir = Path(config.MODELS_DIR)
        subdirs = [p for p in models_dir.iterdir() if p.is_dir() and p.name not in ["saved_models", "exported_models"]]
        if not subdirs:
            print("[INFO] No cached model directories found in models/.")
        for s in subdirs:
            c, b = clean_model_cache(s.name)
            total_files += c
            total_bytes += b
    else:
        target_model = args.model if args.model else config.MODEL_NAME
        c, b = clean_model_cache(target_model)
        total_files += c
        total_bytes += b

    clean_temp_yaml_files()

    total_mb = total_bytes / (1024**2)
    print("\n" + "-"*70)
    print(f"TOTAL RECLAIMED: {total_files:,} files ({total_mb:.2f} MB)".center(70))
    print("-" * 70 + "\n")

if __name__ == "__main__":
    main()
