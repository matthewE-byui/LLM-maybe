@echo off
cd /d "%~dp0"

set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" (
	echo [ERROR] Could not find project Python at: %PY%
	echo Run setup.bat inside llm_project first.
	pause
	exit /b 1
)

REM Run the menu
"%PY%" run_menu.py

pause
