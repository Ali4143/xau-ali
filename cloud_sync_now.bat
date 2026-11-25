@echo off
cd /d "%~dp0"
echo ====================================================
echo   XAUUSD MTF - One-Click Cloud Sync to GitHub
echo ====================================================
python cloud_sync.py
echo.
pause
