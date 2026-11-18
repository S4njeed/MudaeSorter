@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py %*
) else (
    echo Virtual environment .venv not found. Please create it and install requirements.txt.
    exit /b 1
)
