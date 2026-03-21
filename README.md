# Smart Recipe Intelligence System

Professional machine-learning web application for recipe analysis and recommendation.

## What This Project Does

- Health classification for recipes (Healthy, Moderately Healthy, Unhealthy)
- Cuisine discovery using topic modeling
- Recipe cluster similarity insights
- Reinforcement-learning-based cooking strategy recommendation
- Dataset-backed recipe ranking based on user constraints

## Clean Project Layout

```text
MlProject/
|-- app/
|   |-- app.py
|   |-- __init__.py
|   `-- utils/
|       |-- preprocessing.py
|       `-- rl_cooking.py
|-- data/
|   `-- RecipeNLG_dataset.csv
|-- docs/
|   |-- README.md
|   |-- SETUP_GUIDE.md
|   |-- TECHNICAL_SPECIFICATION.md
|   `-- COMPREHENSIVE_DOCUMENTATION.md
|-- models/
|   |-- cuisine_discovery/
|   |-- health_prediction/
|   `-- cooking_optimization/
|-- notebooks/
|   |-- cuisineDiscovery.ipynb
|   `-- healthyPrediction.ipynb
|-- scripts/
|   |-- cooking_optimization_agent.py
|   |-- improve_labels.py
|   `-- verify_models.py
|-- requirements.txt
|-- run_app.ps1
|-- run_app.bat
`-- PROJECT_STRUCTURE.md
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies.
3. Run training notebooks (first time only).
4. Verify model artifacts.
5. Launch the Streamlit app.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/verify_models.py
streamlit run app/app.py
```

Or use:

```powershell
.\run_app.ps1
```

Application URL: http://localhost:8501

## Documentation

- Main docs: docs/README.md
- Setup guide: docs/SETUP_GUIDE.md
- Project structure details: PROJECT_STRUCTURE.md

## Notes

- Keep dataset files in data/.
- Keep trained artifacts in models/.
- Use notebooks/ only for experimentation and training pipelines.
- App entrypoint is app/app.py.
