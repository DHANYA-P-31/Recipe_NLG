# Smart Recipe Intelligence System - PowerShell Launcher
# This script activates the virtual environment and launches the Streamlit app

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Smart Recipe Intelligence System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\.venv\Scripts\Activate.ps1"

# Check if core models exist
if (-not (Test-Path "models\cuisine_discovery\lda_model.pkl") -or -not (Test-Path "models\health_prediction\health_rf_model.pkl")) {
    Write-Host ""
    Write-Host "WARNING: Models not found!" -ForegroundColor Yellow
    Write-Host "Please run the training notebooks first:" -ForegroundColor Yellow
    Write-Host "  1. notebooks/cuisineDiscovery.ipynb" -ForegroundColor Yellow
    Write-Host "  2. notebooks/healthyPrediction.ipynb" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to continue anyway"
}

# Launch Streamlit app using the venv interpreter for reliability
Write-Host ""
Write-Host "Launching Smart Recipe Intelligence System..." -ForegroundColor Green
Write-Host "The app will open in your default browser." -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

& ".\.venv\Scripts\python.exe" -m streamlit run app/app.py
