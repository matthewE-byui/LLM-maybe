@echo off
setlocal

cd /d "%~dp0"

set "PY=%~dp0venv\Scripts\python.exe"
set "SCRIPT=%~dp0train_large_corpus.py"
set "DATA_ROOT=C:\school\LLM Stuff\ChatGPT data 2024-26"
set "CHECKPOINT=%~dp0checkpoints\best_model.pt"
set "LOG_DIR=%~dp0logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "RUN_TS=%DATE%-%TIME%"
set "RUN_TS=%RUN_TS:/=-%"
set "RUN_TS=%RUN_TS::=-%"
set "RUN_TS=%RUN_TS: =0%"
set "LOG_FILE=%LOG_DIR%\overnight-%RUN_TS%.log"

set "MAX_GB=%~1"
if "%MAX_GB%"=="" set "MAX_GB=50"

set "STEPS=%~2"
if "%STEPS%"=="" set "STEPS=10"

echo ================================================== > "%LOG_FILE%"
echo JARVIS Overnight Learning Run >> "%LOG_FILE%"
echo ================================================== >> "%LOG_FILE%"
echo Start: %DATE% %TIME% >> "%LOG_FILE%"
echo Data root: %DATA_ROOT% >> "%LOG_FILE%"
echo Max GB this run: %MAX_GB% >> "%LOG_FILE%"
echo Steps per shard: %STEPS% >> "%LOG_FILE%"
echo Log: %LOG_FILE% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

if not exist "%PY%" (
  echo [ERROR] venv python not found: %PY% >> "%LOG_FILE%"
  exit /b 1
)

if not exist "%SCRIPT%" (
  echo [ERROR] training script not found: %SCRIPT% >> "%LOG_FILE%"
  exit /b 1
)

if not exist "%DATA_ROOT%" (
  echo [ERROR] data root not found: %DATA_ROOT% >> "%LOG_FILE%"
  exit /b 1
)

echo [RUN] Large corpus training... >> "%LOG_FILE%"
"%PY%" "%SCRIPT%" ^
  --data-root "%DATA_ROOT%" ^
  --checkpoint "%CHECKPOINT%" ^
  --max-gb %MAX_GB% ^
  --shard-max-chars 1200000 ^
  --steps %STEPS% ^
  --learning-rate 2e-5 ^
  --min-improvement 0.003 >> "%LOG_FILE%" 2>&1

set "CODE=%ERRORLEVEL%"

echo. >> "%LOG_FILE%"
if "%CODE%"=="0" (
  echo [OK] Overnight learning finished successfully. >> "%LOG_FILE%"
) else (
  echo [FAIL] Overnight learning exited with code %CODE%. >> "%LOG_FILE%"
)

echo End: %DATE% %TIME% >> "%LOG_FILE%"
echo Ledger: %~dp0large_training_ledger.jsonl >> "%LOG_FILE%"

exit /b %CODE%
