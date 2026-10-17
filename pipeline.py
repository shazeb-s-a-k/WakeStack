"""
=============================================================================
                  openWakeWord Custom Model Pipeline Engine
=============================================================================
This script reads `config.py`, automatically configures openWakeWord,
and coordinates the full training, validation, export, and verification flow.

Usage:
    python pipeline.py               # Runs full end-to-end training pipeline
    python pipeline.py --all         # Runs full end-to-end training pipeline
    python pipeline.py --generate    # Stage 1: Generate synthetic audio clips
    python pipeline.py --augment     # Stage 2: Augment clips with acoustic noise
    python pipeline.py --train       # Stage 3: Feature extraction & model training
    python pipeline.py --verify      # Stage 4: Verify trained ONNX model
    python pipeline.py --clean       # Clean intermediate .wav clips to save disk
    python pipeline.py --dry-run     # Validate config & prerequisites only
=============================================================================
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

def print_banner(text, char="="):
    line = char * 70
    print(f"\n{line}\n{text.center(70)}\n{line}\n")

def run_step(step_name, command_args):
    """Executes a subprocess command with clean output and error handling."""
    print(f"\n>>> Running: {step_name}...")
    start_time = time.time()
    
    python_exe = sys.executable
    cmd = [python_exe] + command_args
    
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {result.returncode} (took {elapsed:.1f}s)!")
        sys.exit(result.returncode)
    
    print(f"[OK] {step_name} completed successfully in {elapsed:.1f}s.")
    return elapsed

def preflight_checks():
    """Validates that all required files and datasets are available."""
    print_banner("PRE-FLIGHT ENVIRONMENT & DATASET CHECKS", "-")
    issues = []
    
    # 1. Check ACAV100M features
    acav_path = Path(config.ACAV100M_FEATURES_PATH)
    if not acav_path.exists() or acav_path.stat().st_size < 1000000000:
        issues.append(f"ACAV100M negative features file missing or incomplete: {acav_path}")
    else:
        size_gb = acav_path.stat().st_size / (1024**3)
        print(f"  [x] Negative Dataset: {acav_path.name} ({size_gb:.2f} GB)")

    # 2. Check validation features
    val_path = Path(config.VALIDATION_FEATURES_PATH)
    if not val_path.exists():
        issues.append(f"Validation dataset missing: {val_path}")
    else:
        size_mb = val_path.stat().st_size / (1024**2)
        print(f"  [x] Validation Dataset: {val_path.name} ({size_mb:.2f} MB)")

    # 3. Check Piper generator
    piper_path = Path(config.PIPER_GENERATOR_PATH)
    if not piper_path.exists():
        issues.append(f"Piper sample generator folder not found at: {piper_path}")
    else:
        print(f"  [x] Piper TTS Generator: {piper_path}")

    # 4. Check RIR impulses
    for rir in config.RIR_PATHS:
        rir_path = Path(rir)
        if not rir_path.exists() or not any(rir_path.glob("*.wav")):
            issues.append(f"RIR impulse responses missing in: {rir_path}")
        else:
            impulse_count = len(list(rir_path.glob("*.wav")))
            print(f"  [x] Room Impulse Responses: {impulse_count} filters in {rir_path.name}")

    # 5. Check openWakeWord resources (melspectrogram.onnx, embedding_model.onnx)
    resources_dir = PROJECT_ROOT / "openwakeword" / "openwakeword" / "resources" / "models"
    melspec = resources_dir / "melspectrogram.onnx"
    embed = resources_dir / "embedding_model.onnx"
    if not melspec.exists() or not embed.exists():
        print("  [!] Downloading required openWakeWord base ONNX models...")
        setup_script = PROJECT_ROOT / "setup_resources.py"
        if setup_script.exists():
            subprocess.run([sys.executable, str(setup_script)], check=True)
        else:
            issues.append("Base ONNX models (melspectrogram.onnx / embedding_model.onnx) missing!")
    else:
        print(f"  [x] Base ONNX Audio Encoders verified.")

    # 6. Check Target Phrases and Model Name
    if not config.TARGET_PHRASES or not isinstance(config.TARGET_PHRASES, (list, tuple)):
        issues.append("config.TARGET_PHRASES must be a non-empty list of strings.")
    if not config.MODEL_NAME or not config.MODEL_NAME.isidentifier():
        issues.append(f"config.MODEL_NAME ('{config.MODEL_NAME}') must be a valid identifier (alphanumeric + underscores).")

    print(f"  [x] Target Wake Word: {config.TARGET_PHRASES}")
    print(f"  [x] Model Identifier: {config.MODEL_NAME}")
    print(f"  [x] Training Preset : {config.PRESET} (Steps: {config.STEPS:,}, Samples: {config.N_SAMPLES:,})")

    if issues:
        print("\n[PRE-FLIGHT ERROR] The following issues were found:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    print("\nAll pre-flight checks passed successfully!\n")

def generate_yaml_config():
    """Generates the openWakeWord training YAML configuration from config.py."""
    config_dict = {
        "target_phrase": list(config.TARGET_PHRASES),
        "model_name": config.MODEL_NAME,
        "model_type": config.MODEL_TYPE,
        "layer_size": config.LAYER_SIZE,
        "steps": config.STEPS,
        "custom_negative_phrases": list(config.CUSTOM_NEGATIVE_PHRASES),
        "n_samples": config.N_SAMPLES,
        "n_samples_val": config.N_SAMPLES_VAL,
        "tts_batch_size": config.TTS_BATCH_SIZE,
        "augmentation_rounds": config.AUGMENTATION_ROUNDS,
        "augmentation_batch_size": config.AUGMENTATION_BATCH_SIZE,
        "total_length": 16000,
        "max_negative_weight": config.MAX_NEGATIVE_WEIGHT,
        "target_accuracy": config.TARGET_ACCURACY,
        "target_recall": config.TARGET_RECALL,
        "target_false_positives_per_hour": config.TARGET_FALSE_POSITIVES_PER_HOUR,
        "false_positive_validation_data_path": str(config.VALIDATION_FEATURES_PATH),
        "feature_data_files": {
            "ACAV100M_sample": str(config.ACAV100M_FEATURES_PATH)
        },
        "batch_n_per_class": {
            "ACAV100M_sample": 1024,
            "adversarial_negative": 50,
            "positive": 50
        },
        "output_dir": str(config.MODELS_DIR),
        "piper_sample_generator_path": str(config.PIPER_GENERATOR_PATH),
        "background_paths": list(config.BACKGROUND_PATHS),
        "background_paths_duplication_rate": list(config.BACKGROUND_PATHS_DUPLICATION_RATE),
        "rir_paths": list(config.RIR_PATHS)
    }

    yaml_path = PROJECT_ROOT / f"generated_{config.MODEL_NAME}.yml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"[OK] Training YAML config generated: {yaml_path.name}")
    return yaml_path

def export_and_verify():
    """Copies output models to exported_models/ and verifies ONNX validity."""
    print_banner(f"EXPORT & VERIFICATION FOR '{config.MODEL_NAME}'")
    
    models_dir = Path(config.MODELS_DIR)
    export_dir = Path(config.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_file = models_dir / f"{config.MODEL_NAME}.onnx"
    tflite_file = models_dir / f"{config.MODEL_NAME}.tflite"
    
    if not onnx_file.exists():
        print(f"[ERROR] Trained model file not found: {onnx_file}")
        sys.exit(1)
        
    # Copy to exported_models
    if config.AUTO_EXPORT:
        shutil.copy2(onnx_file, export_dir / onnx_file.name)
        print(f"[OK] Exported ONNX model -> {export_dir / onnx_file.name}")
        if tflite_file.exists():
            shutil.copy2(tflite_file, export_dir / tflite_file.name)
            print(f"[OK] Exported TFLite model -> {export_dir / tflite_file.name}")

    # Verify model
    if config.AUTO_VERIFY:
        verify_script = PROJECT_ROOT / "verify.py"
        if verify_script.exists():
            run_step("Model Verification", [str(verify_script), "--model", str(onnx_file)])
        else:
            print("[INFO] verify.py not found, skipping standalone verification.")

    # Auto clean intermediate clips if configured
    if config.AUTO_CLEAN_CLIPS:
        clean_script = PROJECT_ROOT / "clean_cache.py"
        if clean_script.exists():
            run_step("Cleaning Intermediate Audio Clips", [str(clean_script), "--model", config.MODEL_NAME])

def main():
    parser = argparse.ArgumentParser(description="openWakeWord Training Pipeline Runner")
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline (default)")
    parser.add_argument("--generate", action="store_true", help="Stage 1: Generate synthetic voice clips only")
    parser.add_argument("--augment", action="store_true", help="Stage 2: Augment synthetic clips with noise only")
    parser.add_argument("--train", action="store_true", help="Stage 3: Extract features and train neural network only")
    parser.add_argument("--verify", action="store_true", help="Stage 4: Verify trained ONNX model only")
    parser.add_argument("--clean", action="store_true", help="Clean up intermediate audio clips to save disk space")
    parser.add_argument("--dry-run", action="store_true", help="Validate environment and generate YAML without training")

    args = parser.parse_args()

    # Default to --all if no specific stage flag is set
    run_all = args.all or (not (args.generate or args.augment or args.train or args.verify or args.clean or args.dry_run if hasattr(args, "dry_run") else False))
    if not (args.generate or args.augment or args.train or args.verify or args.clean or args.dry_run):
        run_all = True

    print_banner(f"WAKE WORD TRAINING PIPELINE: {config.MODEL_NAME.upper()}")
    print(f"Target Phrases : {config.TARGET_PHRASES}")
    print(f"Model Preset   : {config.PRESET}")
    print(f"Total Steps    : {config.STEPS:,}")
    print(f"Sample Count   : {config.N_SAMPLES:,} positive clips")

    # Run preflight checks
    preflight_checks()

    # Generate YAML configuration
    yaml_config = generate_yaml_config()

    if args.dry_run:
        print_banner("DRY RUN FINISHED: All configurations and prerequisites are valid!", "*")
        return

    train_script = PROJECT_ROOT / "openwakeword" / "openwakeword" / "train.py"

    total_pipeline_start = time.time()

    # Clean intermediate cache if specifically requested
    if args.clean:
        clean_script = PROJECT_ROOT / "clean_cache.py"
        run_step("Cleaning Cache", [str(clean_script), "--model", config.MODEL_NAME])
        return

    # Stage 1: Generate clips
    if run_all or args.generate:
        print_banner(f"[STAGE 1/4] Generating Synthetic Audio Clips via Piper TTS")
        run_step("Synthetic Voice Generation", [str(train_script), "--training_config", str(yaml_config), "--generate_clips"])

    # Stage 2: Augment clips
    if run_all or args.augment:
        print_banner(f"[STAGE 2/4] Augmenting Audio Clips with Room Impulse Responses & Noise")
        run_step("Acoustic Augmentation", [str(train_script), "--training_config", str(yaml_config), "--augment_clips", "--overwrite"])

    # Stage 3: Train model
    if run_all or args.train:
        print_banner(f"[STAGE 3/4] Neural Network Feature Extraction & Model Training")
        run_step("Model Training & Export", [str(train_script), "--training_config", str(yaml_config), "--train_model"])

    # Stage 4: Export & Verify
    if run_all or args.train or args.verify:
        export_and_verify()

    total_pipeline_time = time.time() - total_pipeline_start
    print_banner(f"PIPELINE FULLY COMPLETE IN {total_pipeline_time/60:.1f} MINUTES!", "*")
    print(f"Saved ONNX Model   : {config.EXPORT_DIR / (config.MODEL_NAME + '.onnx')}")
    print(f"Saved TFLite Model : {config.EXPORT_DIR / (config.MODEL_NAME + '.tflite')}")
    print(f"\nNext Steps:")
    print(f"  1. Test your model on the microphone:")
    print(f"     python test_mic.py --model exported_models/{config.MODEL_NAME}.onnx")
    print(f"  2. (Optional) Reclaim disk space by cleaning intermediate .wav clips:")
    print(f"     python clean_cache.py --model {config.MODEL_NAME}")
    print("=" * 70)

if __name__ == "__main__":
    main()
