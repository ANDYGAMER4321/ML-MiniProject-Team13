@echo off
cd /d "%~dp0"
echo Creating virtual environment in .venv ...
python -m venv .venv || (echo Could not create .venv. Install Python 3.10+ first. & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo Install failed. & pause & exit /b 1)
echo.
echo Setup complete. Double-click run.bat to start the app.
pause
