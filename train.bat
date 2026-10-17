@echo off
title openWakeWord Training Pipeline
color 0A
cls
echo ======================================================================
echo           OPENWAKEWORD CUSTOM MODEL AUTOMATED TRAINING PIPELINE       
echo ======================================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    color 0C
    echo [ERROR] Virtual environment not found in venv\Scripts\activate.bat!
    pause
    exit /b 1
)

echo [1/2] Activating Python Virtual Environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b %errorlevel%
)

echo [2/2] Starting Pipeline Engine (Reading config.py)...
echo.
python pipeline.py %*
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo ======================================================================
    echo [ERROR] Pipeline stopped due to an error. Check logs above.
    echo ======================================================================
    pause
    exit /b %errorlevel%
)

echo.
echo ======================================================================
echo [SUCCESS] Training, Verification, and Export Completed!
echo ======================================================================
echo.
echo You can test your wake word live using:
echo   python test_mic.py
echo.
pause
