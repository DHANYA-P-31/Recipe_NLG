# Comprehensive Documentation

## 1. Purpose

This project provides an end-to-end, local machine-learning recipe intelligence platform with a production-style Streamlit interface.

## 2. Features

- Health prediction with confidence and feature context
- Cuisine discovery with topic probabilities
- Recipe clustering with similarity score
- RL-based cooking strategy recommendation
- Dataset recipe ranking driven by user constraints

## 3. User Flow

1. Launch app (run_app.ps1 or streamlit run app/app.py)
2. Enter recipe title and ingredients
3. Run Analyze Recipe for ML outputs
4. Use RL panel for strategy recommendation and dataset suggestions

## 4. RL Flow (Important)

- RL selects a strategy direction (action)
- Dataset ranking maps that strategy to real recipes
- This is a hybrid approach, not pure RL over the full dataset

## 5. Project Layout

```text
MlProject/
|-- app/
|-- data/
|-- docs/
|-- models/
|-- notebooks/
|-- scripts/
|-- README.md
|-- PROJECT_STRUCTURE.md
|-- run_app.ps1
`-- run_app.bat
```

Detailed structure: ../PROJECT_STRUCTURE.md

## 6. Scripts

- scripts/verify_models.py: verifies required artifacts
- scripts/improve_labels.py: label quality utility
- scripts/cooking_optimization_agent.py: RL training/analysis module

Removed redundant scripts:
- scripts/test_models.py
- scripts/test_rl_dataset.py

## 7. Notebook Policy

notebooks/ contains notebook files only.

Model binaries are stored in models/, not notebooks/.

## 8. Model Training

Run:
- notebooks/cuisineDiscovery.ipynb
- notebooks/healthyPrediction.ipynb

Then verify:

```powershell
python scripts/verify_models.py
```

## 9. Running the App

```powershell
streamlit run app/app.py
```

or

```powershell
.\run_app.ps1
```

## 10. Maintenance Rules

- Keep app paths workspace-relative and consistent
- Keep generated artifacts inside models/
- Keep docs synchronized after structural changes
- Keep tests meaningful and non-duplicate

## 11. Troubleshooting

- Missing model files: run notebooks and verify artifacts
- NLTK download issues: run manual nltk.download commands from setup guide
- Slow recommendation: keep Full Dataset Scan off for quick mode

## 12. Versioning Note

This documentation reflects the cleaned professional structure with consolidated scripts, standardized paths, and updated references.
