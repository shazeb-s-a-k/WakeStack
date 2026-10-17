@echo off
title openWakeWord Training Suite Setup
color 0B
cls
echo ======================================================================
echo             OPENWAKEWORD TRAINING SUITE - 1-CLICK SETUP
echo ======================================================================
echo.

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is not installed or not in your system PATH!
    echo Please install Python 3.9 - 3.11 from python.org and try again.
    pause
    exit /b 1
)

:: 2. Create virtual environment if missing
if not exist "venv\Scripts\activate.bat" (
    echo [1/4] Creating virtual environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment already exists.
)

:: 3. Activate venv
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

:: 4. Install dependencies
echo [3/4] Installing Python dependencies (requirements.txt)...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e openwakeword

:: 5. Download base audio encoders
echo [4/4] Verifying base ONNX models (melspectrogram and embedding)...
python setup_resources.py

:: 6. Check dataset files
echo.
if not exist "openwakeword_features_ACAV100M_2000_hrs_16bit.npy" (
    color 0E
    echo ======================================================================
    echo [NOTICE] The 17 GB ACAV100M training dataset was not found locally.
    echo ======================================================================
    set /p download_now="Would you like to download it now from Hugging Face? [Y/N]: "
    if /i "%download_now%"=="Y" (
        python download_datasets.py
    ) else (
        echo You can download it anytime later by running: python download_datasets.py
    )
) else (
    echo [x] ACAV100M 2,000-hr dataset verified locally!
)

color 0A
echo.
echo ======================================================================
echo [SUCCESS] Environment is fully set up and ready!
echo ======================================================================
echo.
echo Next Steps:
echo   1. Edit `config.py` with your custom wake word.
echo   2. Double-click `train.bat` to train your model.
echo   3. Test live on your mic with `python test_mic.py`.
echo.
pause
