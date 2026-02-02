@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Stopping old processes...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak

echo Updating code from GitHub...
git pull origin main

echo Starting bot...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -u main.py
) else (
    python -u main.py
)
pause
