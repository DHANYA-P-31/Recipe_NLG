# Project Structure

This document defines the current and recommended repository layout.

## Top-Level Structure

```text
MlProject/
|-- app/                    # Streamlit app and reusable logic
|-- data/                   # Datasets (local, large files)
|-- docs/                   # User and technical documentation
|-- models/                 # Trained model artifacts
|-- notebooks/              # Training and exploratory notebooks
|-- scripts/                # Utility and maintenance scripts
|-- requirements.txt        # Python dependencies
|-- run_app.ps1             # PowerShell launcher
|-- run_app.bat             # Windows CMD launcher
|-- README.md               # Project overview
`-- PROJECT_STRUCTURE.md    # This file
```

## app/

```text
app/
|-- app.py                  # Main Streamlit entrypoint
|-- __init__.py
`-- utils/
    |-- preprocessing.py    # Text preprocessing and feature engineering
    |-- rl_cooking.py       # RL environment, agent, and recipe ranking
    `-- __init__.py
```

## models/

```text
models/
|-- cuisine_discovery/      # LDA + KMeans artifacts
|-- health_prediction/      # Health classifiers + preprocessing tools
`-- cooking_optimization/   # Saved RL Q-table policy
```

## scripts/

```text
scripts/
|-- verify_models.py        # Checks required artifacts and app files
|-- improve_labels.py       # Utilities for improving cuisine labels
`-- cooking_optimization_agent.py
```

Removed as redundant:
- scripts/test_models.py
- scripts/test_rl_dataset.py

## notebooks/

```text
notebooks/
|-- cuisineDiscovery.ipynb
`-- healthyPrediction.ipynb
```

Root-level duplicate notebooks were removed to avoid confusion.

## Path Conventions

- App entrypoint: app/app.py
- Dataset path: data/RecipeNLG_dataset.csv
- Model paths: models/<domain>/*.pkl
- Notebook paths: notebooks/*.ipynb

## Operational Flow

1. Train notebooks in notebooks/.
2. Export artifacts to models/.
3. Validate with scripts/verify_models.py.
4. Run app with streamlit run app/app.py or launcher scripts.
