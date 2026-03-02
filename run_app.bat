@echo off
REM Smart Recipe Intelligence System - Launcher
REM This script activates the virtual environment and launches the Streamlit app

echo ========================================
echo Smart Recipe Intelligence System
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if models exist
if not exist "lda_model.pkl" (
    echo.
    echo WARNING: Models not found!
    echo Please run the training notebooks first:
    echo   1. cuisineDiscovery.ipynb
    echo   2. healthyPrediction.ipynb
    echo.
    pause
)

REM Launch Streamlit app
echo.
echo Launching Smart Recipe Intelligence System...
echo The app will open in your default browser.
echo.
echo Press Ctrl+C to stop the server.
echo ========================================
echo.

streamlit run app/app.py

pause
