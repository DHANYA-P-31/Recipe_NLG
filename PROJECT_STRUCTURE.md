# Project Structure Documentation

This document explains the organized folder structure of the Smart Recipe Intelligence System.

## 📂 Directory Overview

```
MlProject/
│
├── 📱 app/                    Application Layer
├── 🤖 models/                 Trained ML Models
├── 📓 notebooks/              Training Notebooks
├── 📊 data/                   Dataset Files
├── ⚙️ scripts/                Utility Scripts
├── 📖 docs/                   Documentation
└── 🔧 Config Files            Root configuration
```

---

## 📱 Application Layer (`app/`)

**Purpose:** Contains the main application code and utilities.

```
app/
├── app.py                    # Main Streamlit application [Entry Point]
└── utils/
    └── preprocessing.py      # Text preprocessing & feature extraction
```

**Key Files:**
- `app.py`: Web interface, model loading, predictions, visualizations
- `preprocessing.py`: Text cleaning, feature engineering, stopwords

---

## 🤖 Models Directory (`models/`)

**Purpose:** Stores all trained machine learning models in organized subdirectories.

### Cuisine Discovery Models

```
models/cuisine_discovery/
├── lda_model.pkl                # LDA topic model (8 topics)
├── count_vectorizer.pkl         # Count vectorizer for LDA
├── lda_labels.pkl               # Cuisine topic labels
├── kmeans_model.pkl             # KMeans clustering model (10 clusters)
├── tfidf_vectorizer.pkl         # TF-IDF vectorizer for clustering
├── kmeans_labels_text.pkl       # Cluster text labels
├── cluster_top_words.pkl        # Top keywords per cluster
├── optimal_lda_k.pkl            # [Optional] Best k for LDA
└── optimal_kmeans_k.pkl         # [Optional] Best k for KMeans
```

**When to Regenerate:**
- After changing dataset
- After modifying preprocessing
- When improving cuisine labels

**Training Notebook:** `notebooks/cuisineDiscovery.ipynb`

### Health Prediction Models

```
models/health_prediction/
├── health_rf_model.pkl          # Random Forest classifier
├── health_svm_model.pkl         # Support Vector Machine classifier
├── health_tfidf_vectorizer.pkl  # TF-IDF vectorizer (5000 features)
└── preprocessing_tools.pkl      # Stopwords, lemmatizer, indicators
```

**When to Regenerate:**
- After changing health indicators
- After updating training data
- When adding new features

**Training Notebook:** `notebooks/healthyPrediction.ipynb`

---

## 📓 Notebooks Directory (`notebooks/`)

**Purpose:** Jupyter notebooks for model training and experimentation.

```
notebooks/
├── cuisineDiscovery.ipynb      # LDA + KMeans training
├── healthyPrediction.ipynb     # Health classification training
└── recipeClustering.ipynb      # Additional clustering experiments
```

**Usage:**
1. Run `cuisineDiscovery.ipynb` to generate cuisine models
2. Run `healthyPrediction.ipynb` to generate health models
3. Run `recipeClustering.ipynb` for experimental analysis

**Note:** These notebooks must be run at least once before launching the app.

---

## 📊 Data Directory (`data/`)

**Purpose:** Contains the dataset used for training.

```
data/
└── RecipeNLG_dataset.csv       # Recipe dataset (2M+ recipes)
```

**Dataset Info:**
- **Source:** RecipeNLG
- **Columns:** title, ingredients, directions, NER
- **Size:** ~1GB
- **Samples:** 2,000,000+ recipes

**Note:** Large file not tracked in git (see `.gitignore`)

---

## ⚙️ Scripts Directory (`scripts/`)

**Purpose:** Utility scripts for maintenance and verification.

```
scripts/
├── verify_models.py            # Check all models exist
├── test_models.py              # Test model loading works
└── improve_labels.py           # Improve cuisine labels
```

**Usage:**

```powershell
# Verify all models are present
python scripts/verify_models.py

# Test that models load without errors
python scripts/test_models.py

# Analyze topics and create better labels
python scripts/improve_labels.py
```

---

## 📖 Documentation Directory (`docs/`)

**Purpose:** Detailed documentation and guides.

```
docs/
├── README.md                   # Comprehensive documentation
└── SETUP_GUIDE.md              # Step-by-step setup instructions
```

**Contents:**
- **README.md:** Full project documentation with examples
- **SETUP_GUIDE.md:** Installation and troubleshooting guide

---

## 🔧 Root Configuration Files

**Purpose:** Project-level configuration and launcher scripts.

```
Root/
├── .gitignore                  # Git ignore rules
├── .venv/                      # Virtual environment (not tracked)
├── requirements.txt            # Python dependencies
├── run_app.ps1                 # PowerShell launcher
├── run_app.bat                 # Batch launcher
└── README.md                   # Quick start guide
```

### Key Files:

**`.gitignore`**
- Excludes: `.venv/`, `*.pyc`, `__pycache__/`, dataset
- Includes: Models (for deployment)

**`requirements.txt`**
- All Python dependencies with versions
- Install: `pip install -r requirements.txt`

**Launch Scripts**
- `run_app.ps1`: PowerShell (recommended)
- `run_app.bat`: Command Prompt
- Both activate venv and start Streamlit

---

## 🔄 File Dependencies

### Model Training Flow

```
Data (data/) 
    → Notebooks (notebooks/)
        → Models (models/)
            → App (app/)
```

### Application Runtime Flow

```
User Request
    → app/app.py
        → models/* (load models)
        → app/utils/preprocessing.py
            → Predictions
                → Streamlit UI
```

---

## 📦 Model File Sizes

Approximate sizes:

| Model | Size | Description |
|-------|------|-------------|
| `lda_model.pkl` | ~10 MB | LDA topic model |
| `kmeans_model.pkl` | ~5 MB | KMeans clustering |
| `health_rf_model.pkl` | ~50 MB | Random Forest (100 trees) |
| `health_svm_model.pkl` | ~20 MB | Support Vector Machine |
| `*_vectorizer.pkl` | ~5-15 MB each | Vectorizers |
| **Total** | ~120 MB | All models combined |

---

## 🔑 Important Paths in Code

### In `app/app.py`:
```python
models['lda_model'] = joblib.load('models/cuisine_discovery/lda_model.pkl')
models['health_rf_model'] = joblib.load('models/health_prediction/health_rf_model.pkl')
```

### In Scripts:
```python
check_file("models/cuisine_discovery/lda_model.pkl", "LDA Model")
check_file("app/utils/preprocessing.py", "Preprocessing Module")
```

---

## 🧹 Cleanup Commands

If you need to reorganize or clean up:

```powershell
# Remove all model files (be careful!)
Remove-Item -Recurse -Force models/

# Clean Python cache
Get-ChildItem -Recurse __pycache__ | Remove-Item -Recurse -Force

# Remove virtual environment
Remove-Item -Recurse -Force .venv/
```

---

## 📊 Model Regeneration Guide

### To Retrain Cuisine Discovery:
1. Open `notebooks/cuisineDiscovery.ipynb`
2. Run all cells
3. Models saved to `models/cuisine_discovery/`
4. Restart app: `streamlit run app/app.py`

### To Retrain Health Prediction:
1. Open `notebooks/healthyPrediction.ipynb`
2. Run all cells
3. Models saved to `models/health_prediction/`
4. Restart app: `streamlit run app/app.py`

---

## 🎯 Best Practices

1. **Never commit `data/`** - Too large for git
2. **Keep models in `models/`** - Organized by purpose
3. **Update docs** when changing structure
4. **Test after changes:** `python scripts/verify_models.py`
5. **Use virtual environment** - Avoid dependency conflicts

---

## 📞 Support

If folder structure issues occur:
1. Run `python scripts/verify_models.py`
2. Check all paths in `app/app.py`
3. Ensure notebooks have been run
4. Review this document

---

**Last Updated:** February 15, 2026  
**Structure Version:** 2.0 (Reorganized)
