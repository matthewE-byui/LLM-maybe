@echo off
REM One-click launcher for Self-Learning AI Chat

cd /d "%~dp0"

set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" (
	echo [ERROR] Could not find project Python at: %PY%
	echo Run setup.bat inside llm_project first.
	pause
	exit /b 1
)

REM Run self-learning chat with default checkpoint
"%PY%" chat_self_learning.py checkpoints/best_model.pt

pause
