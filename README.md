# 🍳 Smart Recipe Intelligence System

A comprehensive ML-powered web application for intelligent recipe analysis and discovery.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🚀 Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Train models (first time only)
# Open and run: notebooks/cuisineDiscovery.ipynb
# Open and run: notebooks/healthyPrediction.ipynb

# Launch application
streamlit run app/app.py
# Or use: .\run_app.ps1
```

**Access at:** http://localhost:8501

---

## 📁 Project Structure

```
MlProject/
│
├── app/                              # Application code
│   ├── app.py                        # Main Streamlit application
│   └── utils/                        # Utility modules
│       └── preprocessing.py          # Text preprocessing functions
│
├── models/                           # Trained ML models
│   ├── cuisine_discovery/            # Cuisine analysis models
│   │   ├── lda_model.pkl            # LDA topic model
│   │   ├── count_vectorizer.pkl     # Count vectorizer
│   │   ├── lda_labels.pkl           # Cuisine labels
│   │   ├── kmeans_model.pkl         # KMeans clustering
│   │   ├── tfidf_vectorizer.pkl     # TF-IDF vectorizer
│   │   ├── kmeans_labels_text.pkl   # Cluster labels
│   │   └── cluster_top_words.pkl    # Cluster keywords
│   │
│   └── health_prediction/            # Health classification models
│       ├── health_rf_model.pkl      # Random Forest classifier
│       ├── health_svm_model.pkl     # SVM classifier
│       ├── health_tfidf_vectorizer.pkl  # TF-IDF vectorizer
│       └── preprocessing_tools.pkl  # Preprocessing utilities
│
├── notebooks/                        # Training notebooks
│   ├── cuisineDiscovery.ipynb       # Cuisine discovery training
│   ├── healthyPrediction.ipynb      # Health classification training
│   └── recipeClustering.ipynb       # Additional clustering experiments
│
├── data/                             # Dataset folder
│   └── RecipeNLG_dataset.csv        # Recipe dataset
│
├── scripts/                          # Utility scripts
│   ├── verify_models.py             # Verify all models exist
│   ├── test_models.py               # Test model loading
│   └── improve_labels.py            # Improve cuisine labels
│
├── docs/                             # Documentation
│   ├── README.md                    # Detailed documentation
│   └── SETUP_GUIDE.md               # Setup instructions
│
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies
├── run_app.ps1                       # PowerShell launcher
└── run_app.bat                       # Batch launcher
```

---

## 🌟 Features

### 🏥 Health Classification
- **Models:** Random Forest & SVM
- **Categories:** Healthy, Moderately Healthy, Unhealthy
- **Features:** 5000+ TF-IDF features + cooking methods
- **Accuracy:** ~85% on test set

### 🌎 Cuisine Discovery
- **Method:** LDA Topic Modeling
- **Topics:** 8 cuisine categories
- **Keywords:** Top characteristic ingredients per cuisine

### 📊 Recipe Clustering
- **Method:** KMeans Clustering
- **Clusters:** 10 recipe categories
- **Similarity:** Cosine distance-based

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Steps

1. **Clone/Download the project**

2. **Create virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Train models** (first time only)
   - Open `notebooks/cuisineDiscovery.ipynb` and run all cells
   - Open `notebooks/healthyPrediction.ipynb` and run all cells

5. **Verify models**
   ```powershell
   python scripts/verify_models.py
   ```

6. **Launch application**
   ```powershell
   streamlit run app/app.py
   ```

---

## 📖 Usage

1. **Open browser** at http://localhost:8501
2. **Input recipe** (title, ingredients, directions)
3. **Click** "Analyze Recipe"
4. **View results** across 3 tabs:
   - Health Analysis
   - Cuisine Discovery
   - Cluster Insights

---

## 🧪 Testing

```powershell
# Verify all models are present
python scripts/verify_models.py

# Test model loading
python scripts/test_models.py
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 85.2% | 84.8% | 85.1% | 84.9% |
| SVM | 83.7% | 83.2% | 83.5% | 83.3% |

**LDA Topics:** 8 discovered cuisine patterns  
**KMeans Clusters:** 10 recipe categories

---

## 🔧 Configuration

### Change Model Paths
Edit `app/app.py` line 50-68 to modify model loading paths.

### Improve Cuisine Labels
```powershell
python scripts/improve_labels.py
```

### Add Features
Modify `app/utils/preprocessing.py` to add new feature extraction logic.

---

## 📚 Documentation

- **[Detailed README](docs/README.md)** - Full documentation
- **[Setup Guide](docs/SETUP_GUIDE.md)** - Step-by-step setup
- **Notebooks** - Training process documentation

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add more cuisine categories
- Improve health scoring
- Add nutritional API integration
- Create recipe recommendations
- Add user profiles

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Dataset:** RecipeNLG
- **Libraries:** scikit-learn, NLTK, Streamlit, Plotly
- **Inspiration:** Food science & ML research

---

## 📧 Support

For issues:
1. Check `docs/SETUP_GUIDE.md`
2. Run `python scripts/verify_models.py`
3. Ensure all dependencies installed

---

**Built with ❤️ using Python, Streamlit, and Machine Learning**
