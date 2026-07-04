@echo off
REM Quick setup script for Windows

echo ================================================
echo LLM Project Setup
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Installing dependencies...
pip install -r requirements.txt

echo.
echo ================================================
echo Setup complete!
echo ================================================
echo.
echo Next steps:
echo   1. Verify setup: python verify_setup.py
echo   2. Train model: python train.py
echo   3. Chat with model: python inference.py
echo.
pause
