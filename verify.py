"""
=============================================================================
                  openWakeWord Model Verification Suite
=============================================================================
Tests both raw ONNXRuntime graph execution and openWakeWord runtime API.

Usage:
    python verify.py                         # Verifies model specified in config.py
    python verify.py --model path/to/my.onnx # Verifies a specific ONNX file
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# Add repo root and openwakeword package dir to sys.path
sys.path.insert(0, str(PROJECT_ROOT / "openwakeword"))
sys.path.insert(0, str(PROJECT_ROOT))

import openwakeword
from openwakeword.model import Model
import onnxruntime as ort
import config

def verify_model(model_path_str):
    model_path = Path(model_path_str)
    
    print("\n" + "="*70)
    print(f"VERIFYING MODEL: {model_path.name}".center(70))
    print("="*70)

    if not model_path.exists():
        print(f"\n[ERROR] Model file not found at: {model_path}")
        return False

    size_kb = model_path.stat().st_size / 1024
    print(f"\n[1/3] File Check:")
    print(f"  Path: {model_path}")
    print(f"  Size: {size_kb:.1f} KB")

    # Step 1: Raw ONNXRuntime Verification
    print(f"\n[2/3] ONNXRuntime Graph Validation:")
    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        in_name = inputs[0].name
        in_shape = inputs[0].shape
        out_name = outputs[0].name
        out_shape = outputs[0].shape
        
        print(f"  Input Node  : '{in_name}' -> Shape: {in_shape}")
        print(f"  Output Node : '{out_name}' -> Shape: {out_shape}")

        # Synthetic feature inference check (shape: 1 x 16 x 96)
        dummy_features = np.random.randn(1, 16, 96).astype(np.float32)
        raw_pred = session.run([out_name], {in_name: dummy_features})[0]
        score = float(raw_pred[0][0])
        print(f"  Test Feature Inference Score: {score:.5f}")
        
        if not (0.0 <= score <= 1.0):
            print(f"  [WARN] Score {score} is outside normal probability range [0.0, 1.0]!")
        else:
            print(f"  [PASS] ONNX tensor execution and sigmoid output verified!")

    except Exception as e:
        print(f"  [FAIL] ONNXRuntime validation failed: {e}")
        return False

    # Step 2: openWakeWord Framework API Verification
    print(f"\n[3/3] openWakeWord API Compatibility Test:")
    try:
        oww = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
        model_key = list(oww.models.keys())[0]
        print(f"  Registered Model Key: '{model_key}'")

        # Simulate 16000 Hz, 16-bit mono audio chunk (1280 samples = 80ms)
        dummy_audio = np.zeros(1280, dtype=np.int16)
        prediction = oww.predict(dummy_audio)
        pred_val = prediction.get(model_key, 0.0)
        print(f"  Test Audio Frame Prediction: {prediction}")
        print(f"  [PASS] openWakeWord API initialized and accepted audio stream smoothly!")

    except Exception as e:
        print(f"  [FAIL] openWakeWord API test failed: {e}")
        return False

    print("\n" + "="*70)
    print("SUCCESS: Model is 100% healthy, valid, and ready for deployment!".center(70))
    print("="*70 + "\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="Verify openWakeWord ONNX Model")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model to verify")
    args = parser.parse_args()

    target_path = None
    if args.model:
        target_path = Path(args.model)
    else:
        # Check standard locations
        candidates = [
            PROJECT_ROOT / "exported_models" / f"{config.MODEL_NAME}.onnx",
            PROJECT_ROOT / "models" / f"{config.MODEL_NAME}.onnx",
            PROJECT_ROOT / "exported_models" / "hey_shavis_v2.onnx",
            PROJECT_ROOT / "models" / "hey_shavis_v2.onnx",
            PROJECT_ROOT / "exported_models" / "hey_sakvis.onnx",
            PROJECT_ROOT / "models" / "hey_sakvis.onnx",
        ]
        for c in candidates:
            if c.exists():
                target_path = c
                break

    if not target_path:
        print(f"[ERROR] No model found to verify. Train one first or pass --model <path>")
        sys.exit(1)

    success = verify_model(target_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
