@echo off
setlocal enabledelayedexpansion
title WakeStack Disk & Cache Cleaner
color 0E
cls
echo ======================================================================
echo                  WAKESTACK CACHE & DISK CLEANER
echo ======================================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    color 0C
    echo [ERROR] Virtual environment not found in venv\Scripts\activate.bat!
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:: If an argument was passed directly, run it immediately without prompt
if not "%~1"=="" (
    if "%~1"=="1" (
        python -u clean_cache.py --all
    ) else if "%~1"=="2" (
        python -u clean_cache.py --datasets
    ) else if "%~1"=="3" (
        python -u clean_cache.py --full
    ) else (
        python -u clean_cache.py %*
    )
    echo.
    pause
    exit /b %errorlevel%
)

:: Interactive Menu
echo What would you like to clean?
echo.
echo   [1] Intermediate Training Cache only (~2-5 GB per model)
echo       - Deletes generated .wav voice clips, noise augmentations, and cache .npy
echo       - PRESERVES your trained models (in exported_models/) and datasets
echo.
echo   [2] Setup Downloaded Datasets only (~17.5 GB)
echo       - Deletes ACAV100M 2,000h & validation .npy files downloaded during setup
echo       - Can be re-downloaded anytime with 'python download_datasets.py'
echo.
echo   [3] Everything (Intermediate Cache + Downloaded Datasets + Base Encoders)
echo       - Reclaims ALL space (~20+ GB)
echo       - PRESERVES your trained models in exported_models/
echo.
echo   [4] Cancel / Exit
echo.
echo ======================================================================
set /p choice="Enter your choice [1-4] (default is 1): "

if "%choice%"=="" set choice=1
if "%choice%"=="1" (
    python -u clean_cache.py --all
) else if "%choice%"=="2" (
    python -u clean_cache.py --datasets
) else if "%choice%"=="3" (
    python -u clean_cache.py --full
) else if "%choice%"=="4" (
    echo [INFO] Cleaner cancelled by user.
    exit /b 0
) else (
    echo [WARNING] Invalid selection "%choice%". Defaulting to intermediate cache:
    python -u clean_cache.py --all
)

echo.
pause
