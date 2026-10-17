"""
=============================================================================
                  Live Microphone Wake Word Tester
=============================================================================
Streams 16kHz audio directly from your microphone into openWakeWord to test
your trained model in real-time with visual score feedback.

Usage:
    python test_mic.py                       # Tests model defined in config.py
    python test_mic.py --model path/to/model.onnx
    python test_mic.py --threshold 0.5
=============================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "openwakeword"))
sys.path.insert(0, str(PROJECT_ROOT))

import openwakeword
from openwakeword.model import Model
import sounddevice as sd
import config

def main():
    parser = argparse.ArgumentParser(description="Real-Time Microphone Wake Word Tester")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model file to test")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0 to 1.0, default 0.5)")
    args = parser.parse_args()

    # Determine target model
    target_path = None
    if args.model:
        target_path = Path(args.model)
    else:
        candidates = [
            PROJECT_ROOT / "exported_models" / f"{config.MODEL_NAME}.onnx",
            PROJECT_ROOT / "models" / f"{config.MODEL_NAME}.onnx",
            PROJECT_ROOT / "exported_models" / "hey_shavis_v2.onnx",
            PROJECT_ROOT / "models" / "hey_shavis_v2.onnx",
            PROJECT_ROOT / "exported_models" / "hey_sakvis.onnx",
        ]
        for c in candidates:
            if c.exists():
                target_path = c
                break

    if not target_path or not target_path.exists():
        print(f"[ERROR] Could not find any ONNX model to test. Please specify with --model <path>.")
        sys.exit(1)

    print("\n" + "="*70)
    print("LIVE MICROPHONE WAKE WORD DETECTOR".center(70))
    print("="*70)
    print(f"Loading model : {target_path.name}")
    print(f"Threshold     : {args.threshold:.2f}")

    try:
        oww = Model(wakeword_models=[str(target_path)], inference_framework="onnx")
        model_key = list(oww.models.keys())[0]
    except Exception as e:
        print(f"[ERROR] Failed to load model with openWakeWord: {e}")
        sys.exit(1)

    # Audio stream parameters: openWakeWord expects 1280 samples per step (80 ms @ 16 kHz)
    CHUNK_SIZE = 1280
    SAMPLE_RATE = 16000

    print(f"Model Key     : '{model_key}'")
    print(f"Sample Rate   : {SAMPLE_RATE} Hz (16-bit Mono, chunk={CHUNK_SIZE})")
    print(f"\nStatus: LISTENING... Speak your wake word! (Press Ctrl+C to quit)\n")
    print("-" * 70)

    last_detect_time = 0
    cooldown_seconds = 2.0

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=CHUNK_SIZE) as stream:
            while True:
                audio_data, overflowed = stream.read(CHUNK_SIZE)
                if overflowed:
                    pass

                # Convert buffer to 1D int16 array
                audio_frame = audio_data.flatten()
                
                # Audio volume calculation (RMS)
                rms = np.sqrt(np.mean(audio_frame.astype(np.float32)**2))
                vol_bars = min(10, int(rms / 500))
                vol_meter = "#" * vol_bars + " " * (10 - vol_bars)

                # Feed chunk to openWakeWord
                prediction = oww.predict(audio_frame)
                score = prediction.get(model_key, 0.0)

                # Visual meter for detection score (20 blocks)
                score_bars = int(score * 20)
                meter = "=" * score_bars + " " * (20 - score_bars)

                now = time.time()
                detected = score >= args.threshold

                if detected and (now - last_detect_time > cooldown_seconds):
                    last_detect_time = now
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"\rVol: [{vol_meter}] | Score: [{meter}] {score:.2f}  <=== *** WAKE WORD DETECTED! *** [{timestamp}]")
                else:
                    # Print live updating meter
                    sys.stdout.write(f"\rVol: [{vol_meter}] | Score: [{meter}] {score:.2f} ")
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[INFO] Stopped listening. Goodbye!")
    except Exception as e:
        print(f"\n[ERROR] Audio stream encountered an error: {e}")

if __name__ == "__main__":
    main()
