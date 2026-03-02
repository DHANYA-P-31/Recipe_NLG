# Quick Setup Guide - Smart Recipe Intelligence System

## 🚀 First-Time Setup Instructions

Follow these steps to set up the Smart Recipe Intelligence System on your machine.

---

## Step 1: Verify Python Installation

Open PowerShell and check your Python version:

```powershell
python --version
```

**Required:** Python 3.8 or higher

If Python is not installed, download from: https://www.python.org/downloads/

---

## Step 2: Create Virtual Environment

Navigate to the project directory and create a virtual environment:

```powershell
cd C:\CODING\PYTHON\MlProject
python -m venv .venv
```

---

## Step 3: Activate Virtual Environment

### Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

You should see `(.venv)` prefix in your terminal.

---

## Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install:
- streamlit
- scikit-learn
- pandas, numpy
- nltk
- plotly
- And more...

**Note:** Installation may take 2-5 minutes.

---

## Step 5: Train the Models (REQUIRED)

### 5.1: Open Jupyter Lab or VS Code

#### Option A: Jupyter Lab
```powershell
pip install jupyterlab
jupyter lab
```

#### Option B: VS Code
Open the project folder in VS Code with Jupyter extension installed.

### 5.2: Run cuisineDiscovery.ipynb

1. Open `cuisineDiscovery.ipynb`
2. Run all cells (Cell → Run All)
3. Wait for completion (~5-10 minutes depending on your machine)
4. Verify these files are created:
   - `lda_model.pkl`
   - `count_vectorizer.pkl`
   - `lda_labels.pkl`
   - `kmeans_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `kmeans_labels_text.pkl`
   - `cluster_top_words.pkl`

### 5.3: Run healthyPrediction.ipynb

1. Open `healthyPrediction.ipynb`
2. Run all cells (Cell → Run All)
3. Wait for completion (~5-10 minutes)
4. Verify these files/folders are created:
   - `models/health_rf_model.pkl`
   - `models/health_svm_model.pkl`
   - `vectorizers/health_tfidf_vectorizer.pkl`
   - `vectorizers/preprocessing_tools.pkl`
   - `models/feature_importance_df.pkl`

---

## Step 6: Launch the Web Application

### Option A: Using Launch Script (Recommended)

#### PowerShell:
```powershell
.\run_app.ps1
```

#### Command Prompt:
```cmd
run_app.bat
```

### Option B: Direct Command

```powershell
streamlit run app.py
```

---

## Step 7: Access the Application

The application will automatically open in your default browser at:

```
http://localhost:8501
```

If it doesn't open automatically, manually navigate to the URL above.

---

## ✅ Verification Checklist

Before launching the app, ensure:

- [ ] Python 3.8+ is installed
- [ ] Virtual environment is created and activated
- [ ] All dependencies are installed (`pip install -r requirements.txt`)
- [ ] `cuisineDiscovery.ipynb` has been run completely
- [ ] `healthyPrediction.ipynb` has been run completely
- [ ] All `.pkl` model files exist in the project directory
- [ ] `models/` and `vectorizers/` folders contain model files

---

## 🔧 Troubleshooting

### Issue: "Execution Policy" Error (PowerShell)

**Error:** `cannot be loaded because running scripts is disabled`

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "Module not found" Error

**Solution:**
```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Models Not Found

**Solution:**
Run both training notebooks completely:
1. `cuisineDiscovery.ipynb` (all cells)
2. `healthyPrediction.ipynb` (all cells, including the new Step 13)

### Issue: NLTK Data Download Failed

**Solution:**
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Issue: Port Already in Use

**Solution:**
```powershell
streamlit run app.py --server.port 8502
```

### Issue: Dataset File Too Large

The `RecipeNLG_dataset.csv` file is large. The notebooks are configured to sample the data for faster training. If you encounter memory issues, reduce the sample sizes in the notebooks.

---

## 📁 Expected File Structure After Setup

```
MlProject/
│
├── app.py                          ✅
├── requirements.txt                ✅
├── README.md                       ✅
├── SETUP_GUIDE.md                  ✅
├── run_app.ps1                     ✅
├── run_app.bat                     ✅
│
├── .venv/                          ✅ (created in Step 2)
│
├── utils/
│   └── preprocessing.py            ✅
│
├── models/                         ✅ (created by healthyPrediction.ipynb)
│   ├── health_rf_model.pkl
│   ├── health_svm_model.pkl
│   └── feature_importance_df.pkl
│
├── vectorizers/                    ✅ (created by healthyPrediction.ipynb)
│   ├── health_tfidf_vectorizer.pkl
│   ├── preprocessing_tools.pkl
│   └── structured_features_info.pkl
│
├── lda_model.pkl                   ✅ (created by cuisineDiscovery.ipynb)
├── count_vectorizer.pkl            ✅
├── lda_labels.pkl                  ✅
├── kmeans_model.pkl                ✅
├── tfidf_vectorizer.pkl            ✅
├── kmeans_labels_text.pkl          ✅
├── cluster_top_words.pkl           ✅
│
├── cuisineDiscovery.ipynb          ✅
├── healthyPrediction.ipynb         ✅
└── RecipeNLG_dataset.csv           ✅
```

---

## 🎉 Success!

If you see the Streamlit app running and can input recipes, you're all set!

Try the example recipe button to see the system in action.

---

## 📞 Need Help?

Refer to:
1. **README.md** - Comprehensive documentation
2. **Troubleshooting section** above
3. Verify all checklist items are completed

---

**Happy Recipe Analyzing! 🍳**
