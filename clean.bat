@echo off
title openWakeWord Cache Cleaner
color 0E
cls
echo ======================================================================
echo                 OPENWAKEWORD INTERMEDIATE CACHE CLEANER
echo ======================================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    color 0C
    echo [ERROR] Virtual environment not found in venv\Scripts\activate.bat!
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python clean_cache.py --all
echo.
pause
