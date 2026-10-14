@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (echo Run setup.bat first. & pause & exit /b 1)
call .venv\Scripts\activate.bat
echo Starting at http://127.0.0.1:5000  (press Ctrl+C to stop)
python app.py
pause
