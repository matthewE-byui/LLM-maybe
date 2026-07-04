@echo off
setlocal

cd /d "%~dp0"

set "PY=%~dp0venv\Scripts\python.exe"
set "SCRIPT=%~dp0autopilot_daemon.py"
set "LOG_DIR=%~dp0logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "CYCLE_MIN=%~1"
if "%CYCLE_MIN%"=="" set "CYCLE_MIN=20"

set "MAX_CYCLES=%~2"
if "%MAX_CYCLES%"=="" set "MAX_CYCLES=0"

set "FORCE_EVERY=%~3"
if "%FORCE_EVERY%"=="" set "FORCE_EVERY=18"

set "RUN_TS=%DATE%-%TIME%"
set "RUN_TS=%RUN_TS:/=-%"
set "RUN_TS=%RUN_TS::=-%"
set "RUN_TS=%RUN_TS: =0%"
set "LOG_FILE=%LOG_DIR%\autopilot-%RUN_TS%.jsonl"

if not exist "%PY%" (
  echo [ERROR] venv python not found: %PY%
  exit /b 1
)

if not exist "%SCRIPT%" (
  echo [ERROR] script not found: %SCRIPT%
  exit /b 1
)

echo Starting unattended autopilot learning...
echo cycle_minutes=%CYCLE_MIN% max_cycles=%MAX_CYCLES% force_maintenance_every=%FORCE_EVERY%
echo log_file=%LOG_FILE%

"%PY%" "%SCRIPT%" --cycle-minutes %CYCLE_MIN% --max-cycles %MAX_CYCLES% --force-maintenance-every %FORCE_EVERY% --log-path "%LOG_FILE%"

exit /b %ERRORLEVEL%
