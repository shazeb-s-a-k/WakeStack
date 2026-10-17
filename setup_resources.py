"""
=============================================================================
                  openWakeWord Resource & Model Setup
=============================================================================
Ensures that all base audio encoders (ONNX) and Piper TTS neural voice
checkpoints (.pt) exist locally before training begins.
=============================================================================
"""

import os
import sys
import time
import urllib.request
import ssl
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
PROJECT_ROOT = Path(__file__).resolve().parent

RESOURCES = [
    {
        "name": "melspectrogram.onnx",
        "dir": PROJECT_ROOT / "openwakeword" / "openwakeword" / "resources" / "models",
        "url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
        "description": "Audio Mel-spectrogram ONNX Encoder"
    },
    {
        "name": "embedding_model.onnx",
        "dir": PROJECT_ROOT / "openwakeword" / "openwakeword" / "resources" / "models",
        "url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
        "description": "Audio Embedding ONNX Encoder"
    },
    {
        "name": "en-us-libritts-high.pt",
        "dir": PROJECT_ROOT / "piper_sample_generator" / "models",
        "url": "https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt",
        "description": "Piper LibriTTS High-Quality Voice Checkpoint"
    },
    {
        "name": "en-us-libritts-high.pt.json",
        "dir": PROJECT_ROOT / "piper_sample_generator" / "models",
        "url": "https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt.json",
        "description": "Piper LibriTTS Voice Config"
    }
]

class DownloadProgressBar:
    def __init__(self, name):
        self.name = name
        self.start_time = time.time()
        self.last_update = 0

    def __call__(self, block_num, block_size, total_size):
        downloaded = block_num * block_size
        now = time.time()
        if now - self.last_update < 0.5 and (total_size <= 0 or downloaded < total_size):
            return
        self.last_update = now

        percent = min(100.0, downloaded * 100.0 / total_size) if total_size > 0 else 0
        elapsed = now - self.start_time
        speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 0

        bar_len = 25
        filled = int(bar_len * percent / 100)
        bar = "=" * filled + "-" * (bar_len - filled)

        sys.stdout.write(f"\r[{bar}] {percent:5.1f}% | {downloaded_mb:.1f}/{total_mb:.1f} MB | {speed:.1f} MB/s ")
        sys.stdout.flush()

def main():
    print("=" * 70)
    print("CHECKING BASE AUDIO MODELS & TTS CHECKPOINTS".center(70))
    print("=" * 70)

    for item in RESOURCES:
        target_dir = Path(item["dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / item["name"]

        print(f"\nResource: {item['description']}")
        print(f"File    : {filepath.name}")

        if filepath.exists() and filepath.stat().st_size > 1000:
            size_mb = filepath.stat().st_size / (1024**2)
            print(f"[OK] Already exists locally ({size_mb:.2f} MB).")
        else:
            print(f"Downloading from GitHub Releases...")
            progress = DownloadProgressBar(item["name"])
            try:
                urllib.request.urlretrieve(item["url"], str(filepath), reporthook=progress)
                print(f"\n[OK] Successfully saved {filepath.name}!")
            except Exception as e:
                print(f"\n[ERROR] Failed to download {item['name']}: {e}")
                sys.exit(1)

    print("\n" + "-" * 70)
    print("Testing ONNXRuntime graph initialization...")
    try:
        import onnxruntime as ort
        melspec_path = PROJECT_ROOT / "openwakeword" / "openwakeword" / "resources" / "models" / "melspectrogram.onnx"
        embed_path = PROJECT_ROOT / "openwakeword" / "openwakeword" / "resources" / "models" / "embedding_model.onnx"
        _ = ort.InferenceSession(str(melspec_path), providers=["CPUExecutionProvider"])
        _ = ort.InferenceSession(str(embed_path), providers=["CPUExecutionProvider"])
        print("[PASS] Base ONNX encoders loaded into ONNXRuntime with CPU provider.")
    except Exception as e:
        print(f"[FAIL] ONNX verification failed: {e}")
        sys.exit(1)

    print("=" * 70)
    print("All base audio models and TTS resources verified successfully!".center(70))
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
