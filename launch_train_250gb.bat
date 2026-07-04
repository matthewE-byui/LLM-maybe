@echo off
setlocal

cd /d "%~dp0"

set "PY=%~dp0venv\Scripts\python.exe"
set "SCRIPT=%~dp0train_large_corpus.py"
set "DATA_ROOT=C:\school\LLM Stuff\ChatGPT data 2024-26"
set "CHECKPOINT=%~dp0checkpoints\best_model.pt"

echo ==================================================
echo JARVIS 250GB Trainer Launcher
echo ==================================================
echo Project: %~dp0
echo Data root: %DATA_ROOT%
echo Checkpoint: %CHECKPOINT%
echo.

if not exist "%PY%" (
  echo [ERROR] Python venv not found: %PY%
  echo Run setup for the project venv first.
  pause
  exit /b 1
)

if not exist "%SCRIPT%" (
  echo [ERROR] Training script not found: %SCRIPT%
  pause
  exit /b 1
)

if not exist "%DATA_ROOT%" (
  echo [ERROR] Data root not found: %DATA_ROOT%
  pause
  exit /b 1
)

if not exist "%CHECKPOINT%" (
  echo [WARN] Checkpoint not found: %CHECKPOINT%
  echo The script will still try to run and may initialize from current defaults.
)

echo Starting training...
echo.

"%PY%" "%SCRIPT%" ^
  --data-root "%DATA_ROOT%" ^
  --checkpoint "%CHECKPOINT%" ^
  --max-gb 250 ^
  --shard-max-chars 1200000 ^
  --steps 10 ^
  --learning-rate 2e-5 ^
  --min-improvement 0.003

set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" (
  echo [OK] Training finished successfully.
) else (
  echo [FAIL] Training exited with code %CODE%.
)

echo Ledger file: %~dp0large_training_ledger.jsonl
pause
exit /b %CODE%
