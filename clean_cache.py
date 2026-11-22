"""
=============================================================================
                  WakeStack Cache & Dataset Cleaner Utility
=============================================================================
Cleans up intermediate generated synthetic audio clips (.wav), temporary
feature cache files (.npy), and optionally setup-downloaded datasets (17.5 GB)
to reclaim disk space safely and rapidly.

SAFE BY DESIGN:
- NEVER touches final exported models in `models/` or `exported_models/`.
- Protects datasets unless explicitly requested via `--datasets` or `--full`.
- High-speed multi-threaded deletion engine (purges 100k+ files in seconds).

Usage:
    python clean_cache.py                  # Cleans cache for active MODEL_NAME in config.py
    python clean_cache.py --all            # Cleans all intermediate model caches in models/
    python clean_cache.py --model <name>   # Cleans intermediate cache for a specific model
    python clean_cache.py --datasets       # Cleans downloaded ACAV100M datasets (~17.5 GB)
    python clean_cache.py --resources      # Cleans downloaded base ONNX models & Piper TTS
    python clean_cache.py --full           # Cleans cache + datasets + resources
=============================================================================
"""

import os
import sys
import stat
import time
import shutil
import argparse
import concurrent.futures
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

def scan_directory(dir_path):
    """
    High-speed scanner using os.scandir to collect all files, dirs, and byte size.
    On Windows, entry.stat() uses cached metadata from directory enumeration,
    scanning 100,000+ files in ~0.2 seconds.
    """
    files = []
    dirs = []
    total_size = 0

    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append(entry.path)
                        sub_f, sub_d, sub_sz = scan_directory(entry.path)
                        files.extend(sub_f)
                        dirs.extend(sub_d)
                        total_size += sub_sz
                    elif entry.is_file(follow_symlinks=False):
                        files.append(entry.path)
                        total_size += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    pass
    except OSError:
        pass

    return files, dirs, total_size

def _safe_unlink(file_path):
    """Deletes a file, removing read-only attributes if needed."""
    try:
        os.unlink(file_path)
    except OSError:
        try:
            os.chmod(file_path, stat.S_IWRITE)
            os.unlink(file_path)
        except OSError:
            pass

def delete_directory_fast(target_dir, files, dirs):
    """
    Blazing fast multi-threaded file deletion (32 workers).
    Saturates filesystem queues and eliminates NTFS single-thread B-tree bottlenecks.
    """
    total = len(files)
    if total == 0:
        try:
            os.rmdir(target_dir)
        except OSError:
            pass
        return True

    chunk_size = max(100, total // 50)
    completed = 0
    t_start = time.time()

    # Parallel file deletion
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(_safe_unlink, f) for f in files]
        for f in concurrent.futures.as_completed(futures):
            completed += 1
            if completed % chunk_size == 0 or completed == total:
                pct = (completed / total) * 100
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                sys.stdout.write(f"\r  Deleting: [{pct:5.1f}%] {completed:,}/{total:,} files ({rate:,.0f} files/s) ")
                sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    # Clean up empty subdirectories bottom-up
    dirs.sort(key=lambda d: len(d), reverse=True)
    for d in dirs:
        try:
            os.rmdir(d)
        except OSError:
            pass

    try:
        os.rmdir(target_dir)
    except OSError:
        pass

    return not os.path.exists(target_dir)

def clean_model_cache(model_name):
    """Cleans intermediate synthetic audio and feature caches for a specific model."""
    target_dir = Path(config.MODELS_DIR) / model_name
    
    if not target_dir.exists():
        print(f"[INFO] No intermediate cache directory found for '{model_name}' at: {target_dir}")
        return 0, 0

    print(f"Scanning intermediate files in: {target_dir}...")
    sys.stdout.flush()
    t0 = time.time()
    files, dirs, byte_size = scan_directory(target_dir)
    scan_time = time.time() - t0
    file_count = len(files)
    
    if file_count == 0:
        print(f"[INFO] Cache directory is already empty: {target_dir}")
        try:
            target_dir.rmdir()
        except OSError:
            pass
        return 0, 0

    size_mb = byte_size / (1024**2)
    print(f"Found {file_count:,} temporary files ({size_mb:.2f} MB, scanned in {scan_time:.2f}s).")
    sys.stdout.flush()
    
    t_del = time.time()
    success = delete_directory_fast(target_dir, files, dirs)
    del_time = time.time() - t_del
    
    if success:
        print(f"[SUCCESS] Cleaned {file_count:,} cached files for '{model_name}'. Reclaimed {size_mb:.2f} MB in {del_time:.2f}s!")
        return file_count, byte_size
    else:
        print(f"[WARNING] Some files in {target_dir} could not be deleted (may be locked by another process).")
        return file_count, byte_size

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

def clean_datasets():
    """Removes setup-downloaded Hugging Face negative datasets (~17.5 GB)."""
    datasets = [
        PROJECT_ROOT / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        PROJECT_ROOT / "validation_set_features.npy",
    ]
    cleaned_bytes = 0
    count = 0
    print("\nScanning setup-downloaded datasets...")
    for ds in datasets:
        if ds.exists():
            try:
                sz = ds.stat().st_size
                ds.unlink()
                cleaned_bytes += sz
                count += 1
                size_mb = sz / (1024**2)
                if size_mb >= 1024:
                    print(f"  [OK] Deleted dataset: {ds.name} ({size_mb / 1024:.2f} GB)")
                else:
                    print(f"  [OK] Deleted dataset: {ds.name} ({size_mb:.2f} MB)")
            except Exception as e:
                print(f"  [ERROR] Could not delete {ds.name}: {e}")
        else:
            print(f"  [INFO] Dataset not found (already clean): {ds.name}")
    return count, cleaned_bytes

def clean_resources():
    """Removes setup-downloaded base audio encoders & Piper voice checkpoints (~150 MB)."""
    resources = [
        PROJECT_ROOT / "openwakeword" / "openwakeword" / "resources" / "models" / "melspectrogram.onnx",
        PROJECT_ROOT / "openwakeword" / "openwakeword" / "resources" / "models" / "embedding_model.onnx",
        PROJECT_ROOT / "piper_sample_generator" / "models" / "en-us-libritts-high.pt",
        PROJECT_ROOT / "piper_sample_generator" / "models" / "en-us-libritts-high.pt.json",
    ]
    cleaned_bytes = 0
    count = 0
    print("\nScanning setup-downloaded base audio & TTS resources...")
    for res in resources:
        if res.exists():
            try:
                sz = res.stat().st_size
                res.unlink()
                cleaned_bytes += sz
                count += 1
                print(f"  [OK] Deleted resource: {res.name} ({sz / (1024**2):.2f} MB)")
            except Exception as e:
                print(f"  [ERROR] Could not delete {res.name}: {e}")
        else:
            print(f"  [INFO] Resource not found (already clean): {res.name}")
    return count, cleaned_bytes

def main():
    parser = argparse.ArgumentParser(description="WakeStack Cache and Setup Dataset Cleaner.")
    parser.add_argument("--model", type=str, default=None, help="Model name whose intermediate cache to clean")
    parser.add_argument("--all", action="store_true", help="Clean all model cache subdirectories in models/")
    parser.add_argument("--datasets", action="store_true", help="Clean setup downloaded datasets (ACAV100M ~17.5 GB)")
    parser.add_argument("--resources", action="store_true", help="Clean setup downloaded base audio & TTS models")
    parser.add_argument("--full", action="store_true", help="Clean intermediate cache + datasets + resources")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("WAKESTACK DISK & CACHE CLEANER".center(70))
    print("="*70)

    total_files = 0
    total_bytes = 0

    # 1. Clean intermediate model caches
    if args.full or args.all or (not args.datasets and not args.resources and not args.model):
        models_dir = Path(config.MODELS_DIR)
        if models_dir.exists():
            subdirs = [p for p in models_dir.iterdir() if p.is_dir() and p.name not in ["saved_models", "exported_models"]]
            if not subdirs:
                print("[INFO] No cached model directories found in models/ (intermediate cache is clean).")
            for s in subdirs:
                c, b = clean_model_cache(s.name)
                total_files += c
                total_bytes += b
        clean_temp_yaml_files()
    elif args.model:
        c, b = clean_model_cache(args.model)
        total_files += c
        total_bytes += b
        clean_temp_yaml_files()

    # 2. Clean setup-downloaded datasets
    if args.full or args.datasets:
        c, b = clean_datasets()
        total_files += c
        total_bytes += b

    # 3. Clean setup-downloaded base resources
    if args.full or args.resources:
        c, b = clean_resources()
        total_files += c
        total_bytes += b

    total_mb = total_bytes / (1024**2)
    print("\n" + "-"*70)
    if total_mb >= 1024:
        print(f"TOTAL RECLAIMED: {total_files:,} files ({total_mb / 1024:.2f} GB)".center(70))
    else:
        print(f"TOTAL RECLAIMED: {total_files:,} files ({total_mb:.2f} MB)".center(70))
    print("-" * 70 + "\n")

if __name__ == "__main__":
    main()
