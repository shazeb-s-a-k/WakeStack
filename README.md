<div align="center">

# ⚡ WakeStack

### Lightweight, CPU-Optimized Custom Wake Word Training Suite

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![openWakeWord](https://img.shields.io/badge/openWakeWord-v0.6.0-blue)](https://github.com/dscripka/openWakeWord)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20%26%20CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()

**WakeStack** is a streamlined, CUI/CLI-first automated suite for training high-accuracy, production-grade custom wake words using **openWakeWord** and **Piper TTS**.

*Designed from the ground up to be lightweight, dependable, and accessible on low-end PCs without needing expensive GPU cloud setups or complex machine learning pipelines.*

</div>

---

## 🌟 Why WakeStack?

Most wake word training tools either require heavy cloud subscriptions, break frequently in Google Colab disconnects, or demand multi-gigabyte browser GUIs that starve system RAM during training.

**WakeStack is built for efficiency:**
- 🖥️ **Low-End PC Friendly**: Runs comfortably on standard CPUs (Intel Core i3/i5/i7, AMD Ryzen). No dedicated GPU required!
- ⚡ **Zero Bloat (CUI-First)**: Lightweight console interface saves every megabyte of RAM and CPU cycle for model training.
- 🎯 **Single-File Configuration**: You only ever need to edit `config.py` to train a new wake word.
- 🔁 **1-Click Full Automation (`train.bat`)**: Automatically generates synthetic voices, applies room reverb filters, extracts embeddings, trains neural network classifiers, and exports `.onnx` and `.tflite` models.
- 🎙️ **Live Console Microphone Tester (`test_mic.py`)**: Speak into your microphone and watch live volume & confidence meters in real time.
- 🧹 **1-Click Cache Cleaner (`clean.bat`)**: Safely reclaims gigabytes of temporary training clips without touching your trained models or datasets.

---

## 💻 Minimum System Requirements

WakeStack is engineered to run on modest consumer hardware:

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | Windows 10 / 11 (64-bit) or Linux | Windows 10 / 11 or Ubuntu 20.04+ |
| **Processor (CPU)** | Dual-Core 64-bit CPU | 4+ Cores (Core i5/i7 or AMD Ryzen) |
| **System Memory (RAM)** | 8 GB | 16 GB |
| **Graphics (GPU)** | *None required* (CPU-only supported) | NVIDIA GPU with CUDA (optional) |
| **Free Storage** | ~20 GB (for 2,000h negative dataset) | SSD with 25+ GB free space |
| **Python** | Python 3.9, 3.10, or 3.11 | Python 3.10 |

---

## 🚀 Quickstart Guide

### 1. Clone & Setup
Clone the repository and run the 1-click environment installer:

```bash
git clone https://github.com/shazeb-s-a-k/WakeStack.git
cd WakeStack
```

On Windows, double-click **`setup.bat`** (or run manually):
```cmd
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -e openwakeword
python setup_resources.py
```

### 2. Download Training Datasets (One-Time)
To train models with near-zero false positive rates, WakeStack uses precomputed feature embeddings (2,000 hours of background speech/noise from the ACAV100M dataset):

```cmd
python download_datasets.py
```
*(Automatically downloads `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` and `validation_set_features.npy` directly from Hugging Face with real-time progress bars).*

---

## 🎙️ Training Any Wake Word in 3 Steps

### Step 1: Configure Your Wake Word in `config.py`
Open [`config.py`](config.py) and enter your wake word phrase:

```python
# The wake word phrase(s) to train (phonetic variations improve accuracy)
TARGET_PHRASES = ["hey computer", "computer"]

# Output model name (will generate hey_computer.onnx)
MODEL_NAME = "hey_computer"

# Choose preset:
#   "PRODUCTION" : 400,000 steps, 50,000 samples (~4-8 hrs on CPU, production-grade accuracy)
#   "FAST_TEST"  : 20,000 steps, 5,000 samples (~15-30 mins, quick validation run)
PRESET = "PRODUCTION"
```

### Step 2: Run Training
Double-click **`train.bat`** or run in your terminal:
```cmd
train.bat
```
WakeStack will automatically execute all 4 stages:
1. **Voice Synthesis**: Generates diverse synthetic voice samples using Piper TTS.
2. **Acoustic Augmentation**: Adds room impulse responses (reverb) and background noise.
3. **Model Training**: Trains the DNN classifier on 2,000 hours of negative audio features.
4. **Export & Verification**: Verifies the ONNX execution graph and saves completed models to [`exported_models/`](exported_models/).

### Step 3: Test Live with Your Microphone!
Speak your wake word directly into your microphone to verify detection:
```cmd
venv\Scripts\activate.bat
python test_mic.py
```
You will see a live console meter showing volume and detection probability in real time:
```text
Vol: [#####     ] | Score: [====================] 0.98  <=== *** WAKE WORD DETECTED! *** [14:20:15]
```

---

## 📁 Repository Structure

```
WakeStack/
├── config.py                 # Primary user configuration file
├── train.bat                 # 1-Click training launcher
├── test_mic.py               # Live real-time microphone tester
├── clean.bat                 # 1-Click cache cleaner to reclaim disk space
├── setup.bat                 # 1-Click environment setup script
├── pipeline.py               # Master training engine & orchestrator
├── verify.py                 # ONNXRuntime & openWakeWord model validator
├── clean_cache.py            # Cache cleaner utility (preserves models)
├── setup_resources.py        # Downloader for base audio encoders & Piper TTS
├── download_datasets.py      # Downloader for Hugging Face ACAV100M datasets
├── requirements.txt          # Python dependencies
├── exported_models/          # Your finished .onnx and .tflite models
├── piper_sample_generator/   # Piper TTS synthetic voice engine & impulse filters
└── openwakeword/             # Core openWakeWord framework (Windows patched)
```

---

## 🧹 Reclaiming Disk Space (`clean.bat`)

During synthetic generation, openWakeWord generates temporary `.wav` audio clips (~2 to 5 GB). Once your model is trained and exported to `exported_models/`, these temporary files are no longer needed.

- **Manual**: Double-click **`clean.bat`** anytime to purge intermediate audio clips.
- **Automatic**: Set `AUTO_CLEAN_CLIPS = True` in `config.py` to auto-clean after training completes.

> **Safety Guarantee**: The cleaner will **NEVER** delete your trained `.onnx`/`.tflite` models or the 17.2 GB ACAV100M dataset.

---

## 🤖 Deploying Your Model (SHAVIS, Home Assistant, Python)

Once trained, your ready-to-use model will be in:
`exported_models/<MODEL_NAME>.onnx`

### Python Integration Example:
```python
import numpy as np
from openwakeword.model import Model

# Load your custom model
oww = Model(
    wakeword_models=["exported_models/hey_computer.onnx"],
    inference_framework="onnx"
)

# Feed 16kHz, 16-bit mono audio in 1280-sample chunks (80 ms)
audio_chunk = np.zeros(1280, dtype=np.int16)
predictions = oww.predict(audio_chunk)

if predictions.get("hey_computer", 0.0) >= 0.5:
    print("Wake word detected!")
```

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [openWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka
- [Piper](https://github.com/rhasspy/piper) by Rhasspy / Michael Hansen
