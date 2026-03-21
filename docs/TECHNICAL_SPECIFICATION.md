# Technical Specification

## System Overview

Smart Recipe Intelligence System is a local Streamlit application that combines:

- Health classification (Random Forest, SVM)
- Cuisine discovery (LDA topic model)
- Recipe clustering (KMeans)
- Cooking strategy recommendation (tabular Q-learning)
- Large-scale dataset ranking for practical recipe suggestions

## Runtime Architecture

```text
UI (Streamlit: app/app.py)
  -> Preprocessing (app/utils/preprocessing.py)
  -> Model Inference (models/*)
  -> RL Strategy + Ranking (app/utils/rl_cooking.py)
  -> Visual Output (Plotly + Streamlit tables)
```

## Directory Contracts

- app/: app runtime code
- data/: input datasets (RecipeNLG CSV)
- models/: persisted model artifacts
- notebooks/: training notebooks only
- scripts/: verification and maintenance utilities
- docs/: user and technical documentation

## Model Artifacts

### models/cuisine_discovery/
- lda_model.pkl
- count_vectorizer.pkl
- lda_labels.pkl
- kmeans_model.pkl
- tfidf_vectorizer.pkl
- kmeans_labels_text.pkl
- cluster_top_words.pkl
- optional: optimal_lda_k.pkl, optimal_kmeans_k.pkl

### models/health_prediction/
- health_rf_model.pkl
- health_svm_model.pkl
- health_tfidf_vectorizer.pkl
- preprocessing_tools.pkl
- structured_features_info.pkl
- feature_importance_df.pkl

### models/cooking_optimization/
- q_learning_agent.pkl
- learned_policy_top_states.csv
- training_reward_vs_episode.png

## RL Algorithm Summary

Implementation: tabular Q-learning with epsilon-greedy exploration.

Update rule:

Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))

Defaults:
- alpha: 0.1
- gamma: 0.9
- epsilon: 0.2 decaying to 0.02

## Entry Points

- Main app: app/app.py
- PowerShell launch: run_app.ps1
- CMD launch: run_app.bat
- Artifact verification: scripts/verify_models.py

## Operational Requirements

- Python 3.10+
- Virtual environment recommended
- Local model files available in models/
- data/RecipeNLG_dataset.csv present for dataset ranking mode

## Validation Checklist

1. python scripts/verify_models.py exits successfully
2. streamlit run app/app.py starts without errors
3. RL panel returns a strategy and dataset recommendations
4. All documentation paths resolve to existing files
