"""
=============================================================================
                  Wake Word Training Configuration File
=============================================================================
To train a new custom wake word:
  1. Set TARGET_PHRASES to your desired wake word (e.g., ["hey jarvis"]).
  2. Set MODEL_NAME to your model's identifier (e.g., "hey_jarvis").
  3. (Optional) Adjust PRESET or fine-tune hyperparameters below.
  4. Run `train.bat` or `python pipeline.py`.
=============================================================================
"""

import os
from pathlib import Path

# Base directories (dynamically detected)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
EXPORT_DIR = BASE_DIR / "exported_models"

# ===========================================================================
# 1. CORE WAKE WORD SETTINGS (EDIT THESE FOR NEW WAKE WORDS)
# ===========================================================================

# The target wake word phrase(s) to train on.
# You can provide multiple spelling or phonetic variations for better accuracy.
# Examples:
#   TARGET_PHRASES = ["hey shavis", "hey shahvis"]
#   TARGET_PHRASES = ["hey computer"]
#   TARGET_PHRASES = ["activate assistant"]
TARGET_PHRASES = [
    "hey shazeb",
    "hey shahzeb",
    "hey shazaib",
    "hey shahzaib",
    "hi shahzeb",
    "hi shazeb",
    "hey shahzaib",
]

# Output model name (without file extension).
# Used for the generated .onnx and .tflite files (e.g., hey_shavis.onnx).
MODEL_NAME = "hey_shavis_v3"

# ===========================================================================
# 2. TRAINING PRESETS
# ===========================================================================
# Choose between:
#   "PRODUCTION" : Full training (400k steps, 50k samples) - Best accuracy & lowest false positives (~4-8 hours)
#   "FAST_TEST"  : Quick test run (20k steps, 5k samples)  - Tests the whole pipeline in ~15-30 minutes
#   "CUSTOM"     : Uses the manual hyperparameter overrides defined below
PRESET = "PRODUCTION"

# ===========================================================================
# 3. HYPERPARAMETERS (Used when PRESET == "CUSTOM", or as defaults)
# ===========================================================================
if PRESET == "FAST_TEST":
    STEPS = 20000
    N_SAMPLES = 5000
    N_SAMPLES_VAL = 1000
    AUGMENTATION_ROUNDS = 2
elif PRESET == "PRODUCTION":
    STEPS = 400000
    N_SAMPLES = 50000
    N_SAMPLES_VAL = 5000
    AUGMENTATION_ROUNDS = 5
else:  # "CUSTOM"
    STEPS = 400000
    N_SAMPLES = 50000
    N_SAMPLES_VAL = 5000
    AUGMENTATION_ROUNDS = 5

# Neural Network Architecture
MODEL_TYPE = "dnn"          # "dnn" (recommended) or "rnn"
LAYER_SIZE = 16             # Hidden layer dimension (16 or 32 for ultra-fast low-latency inference)

# Batch sizes for synthetic generation & augmentation
TTS_BATCH_SIZE = 50
AUGMENTATION_BATCH_SIZE = 50

# Custom negative phrases to prevent false activations.
# Phrases that sound somewhat similar to your target phrase, but should NOT trigger the wake word.
# Example for "hey shavis": ["hey sam", "hey charlie", "hey show me", "hey save this"]
CUSTOM_NEGATIVE_PHRASES = [
    "hey show me",
    "hey save this",
    "hey charlie",
    "hey sam",
]

# Optimization and Thresholds
MAX_NEGATIVE_WEIGHT = 1000
TARGET_ACCURACY = 0.6
TARGET_RECALL = 0.25
TARGET_FALSE_POSITIVES_PER_HOUR = 0.2

# ===========================================================================
# 4. DATASET & RESOURCE PATHS
# ===========================================================================
# Pre-computed negative background feature dataset (ACAV100M 2,000 hours)
ACAV100M_FEATURES_PATH = str(BASE_DIR / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")

# Pre-computed false positive validation dataset
VALIDATION_FEATURES_PATH = str(BASE_DIR / "validation_set_features.npy")

# Synthetic voice generator (Piper TTS)
PIPER_GENERATOR_PATH = str(BASE_DIR / "piper_sample_generator")

# Room Impulse Response (RIR) acoustic filters for room reverb simulation
RIR_PATHS = [str(BASE_DIR / "piper_sample_generator" / "impulses")]

# Background noise directory (optional, leave empty if using ACAV100M)
BACKGROUND_PATHS = []
BACKGROUND_PATHS_DUPLICATION_RATE = []

# ===========================================================================
# 5. POST-TRAINING AUTOMATION
# ===========================================================================
# Automatically copy trained .onnx and .tflite models to `exported_models/`
AUTO_EXPORT = True

# Automatically run ONNX verification test after training finishes
AUTO_VERIFY = True

# Automatically clean up intermediate generated .wav audio clips after training
# Set to True if you want to save disk space (~5 GB) automatically once training succeeds
AUTO_CLEAN_CLIPS = False
