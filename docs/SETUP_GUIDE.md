# Setup Guide

## 1. Prerequisites

- Python 3.10+
- Windows PowerShell or Command Prompt
- At least 8 GB RAM recommended for model workflows

## 2. Create Environment

```powershell
cd C:\CODING\PYTHON\MlProject
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Train Models (First Time)

Run both notebooks from the notebooks folder:

1. notebooks/cuisineDiscovery.ipynb
2. notebooks/healthyPrediction.ipynb

These notebooks generate model artifacts under:

- models/cuisine_discovery/
- models/health_prediction/
- models/cooking_optimization/ (created by app on first RL training)

## 4. Verify Artifacts

```powershell
python scripts/verify_models.py
```

## 5. Launch Application

```powershell
streamlit run app/app.py
```

Or launch with scripts:

```powershell
.\run_app.ps1
```

## 6. Common Issues

### PowerShell execution policy

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Missing NLTK assets

```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Port conflict

```powershell
streamlit run app/app.py --server.port 8502
```

## 7. Clean Expected Structure

```text
app/, data/, docs/, models/, notebooks/, scripts/
```

If your structure differs, compare against ../PROJECT_STRUCTURE.md.
