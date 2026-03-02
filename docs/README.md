# 🍳 Smart Recipe Intelligence System

A comprehensive ML-powered web application for intelligent recipe analysis and discovery. This system integrates multiple machine learning models to provide health classification, cuisine discovery, and recipe clustering insights.


---

## 🌟 Features

### 🏥 Health Classification
- **Multi-class prediction**: Healthy, Moderately Healthy, or Unhealthy
- **Dual model support**: Random Forest and SVM classifiers
- **Confidence scoring**: Probability-based predictions
- **Feature importance analysis**: Understand which ingredients influence health predictions
- **Structured feature extraction**: Cooking methods, ingredient counts, and more

### 🌎 Cuisine Discovery
- **LDA Topic Modeling**: Discover hidden cuisine patterns
- **Probabilistic classification**: Multi-topic distribution analysis
- **Top keywords extraction**: Understand characteristic ingredients
- **Visual probability distribution**: Interactive charts showing all cuisine matches

### 📊 Recipe Clustering
- **KMeans clustering**: Group similar recipes together
- **Cluster categorization**: 10+ distinct recipe categories
- **Similarity scoring**: Distance-based confidence metrics
- **Pattern recognition**: Identify recipe styles and themes

---

## 🏗️ Project Structure

```
MlProject/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── utils/
│   └── preprocessing.py            # Preprocessing utilities
│
├── models/                         # Trained ML models (generated)
│   ├── health_rf_model.pkl
│   ├── health_svm_model.pkl
│   └── feature_importance_df.pkl
│
├── vectorizers/                    # Vectorizers & preprocessing tools
│   ├── health_tfidf_vectorizer.pkl
│   ├── preprocessing_tools.pkl
│   └── structured_features_info.pkl
│
├── cuisineDiscovery.ipynb         # Cuisine discovery training notebook
├── healthyPrediction.ipynb        # Health classification training notebook
├── recipeClustering.ipynb         # Recipe clustering notebook
│
├── lda_model.pkl                  # LDA topic model
├── count_vectorizer.pkl           # Count vectorizer for LDA
├── lda_labels.pkl                 # LDA topic labels
├── kmeans_model.pkl               # KMeans clustering model
├── tfidf_vectorizer.pkl           # TF-IDF vectorizer for clustering
├── kmeans_labels_text.pkl         # Cluster labels
└── cluster_top_words.pkl          # Top words per cluster
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone or Download the Project

```bash
cd C:\CODING\PYTHON\MlProject
```

### Step 2: Create Virtual Environment (Recommended)

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Train Models (First Time Only)

**Important:** You must train the models before running the web app.

#### 4.1: Run Cuisine Discovery Notebook
```powershell
# Open and run all cells in:
# cuisineDiscovery.ipynb
```
This will generate:
- `lda_model.pkl`
- `count_vectorizer.pkl`
- `lda_labels.pkl`
- `kmeans_model.pkl`
- `tfidf_vectorizer.pkl`
- `kmeans_labels_text.pkl`
- `cluster_top_words.pkl`

#### 4.2: Run Health Prediction Notebook
```powershell
# Open and run all cells in:
# healthyPrediction.ipynb
```
This will generate:
- `models/health_rf_model.pkl`
- `models/health_svm_model.pkl`
- `vectorizers/health_tfidf_vectorizer.pkl`
- `vectorizers/preprocessing_tools.pkl`
- `models/feature_importance_df.pkl`

### Step 5: Launch the Web Application

```powershell
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Input Your Recipe

1. **Recipe Title**: Enter a descriptive recipe name
2. **Ingredients**: List all ingredients (one per line or comma-separated)
3. **Directions** (Optional): Add cooking instructions for better analysis

### Click "Analyze Recipe"

The system will process your input and provide:

### 🏥 Health Analysis Tab
- **Health Category**: Healthy / Moderately Healthy / Unhealthy
- **Confidence Score**: Model prediction confidence (%)
- **Model Comparison**: Random Forest vs SVM predictions
- **Top Influencing Ingredients**: Features that drove the prediction
- **Recipe Characteristics**: Cooking methods, ingredient counts

### 🌎 Cuisine Discovery Tab
- **Discovered Cuisine Style**: Primary cuisine match
- **Confidence Score**: Topic probability (%)
- **Top Keywords**: Characteristic ingredients and terms
- **Full Distribution**: Probabilities across all cuisine styles

### 📊 Cluster Insights Tab
- **Cluster Category**: Recipe cluster assignment
- **Similarity Score**: How well it matches the cluster
- **Cluster Keywords**: Common ingredients in this cluster
- **Distance Analysis**: Similarity to all clusters

---

## 🎯 Example Recipes to Try

### Example 1: Healthy Recipe
```
Title: Grilled Salmon with Steamed Vegetables
Ingredients:
Fresh salmon fillet
Olive oil
Lemon juice
Garlic cloves
Broccoli florets
Carrot slices
Asparagus
Salt and pepper
Fresh herbs (dill, parsley)

Directions:
Marinate salmon with olive oil, lemon juice, and garlic.
Grill for 5-7 minutes per side.
Steam vegetables until tender.
Season and serve with fresh herbs.
```

### Example 2: Dessert Recipe
```
Title: Chocolate Chip Cookies
Ingredients:
2 cups all-purpose flour
1 cup butter
3/4 cup brown sugar
3/4 cup white sugar
2 eggs
2 tsp vanilla extract
1 tsp baking soda
1/2 tsp salt
2 cups chocolate chips

Directions:
Cream butter and sugars. Add eggs and vanilla.
Mix dry ingredients separately. Combine all.
Fold in chocolate chips. Bake at 375°F for 10-12 minutes.
```

### Example 3: International Cuisine
```
Title: Chicken Tikka Masala
Ingredients:
Chicken breast pieces
Yogurt
Garam masala
Turmeric
Cumin
Ginger garlic paste
Tomato puree
Heavy cream
Butter
Cilantro

Directions:
Marinate chicken in yogurt and spices.
Grill chicken pieces. Prepare curry sauce with tomatoes and cream.
Simmer chicken in sauce. Garnish with cilantro.
```

---

## 🔧 Configuration Options

### Model Selection
- Switch between **Random Forest** and **SVM** for health predictions
- Random Forest provides feature importance analysis
- SVM offers faster predictions

### Customization
You can modify prediction behavior by editing:
- `utils/preprocessing.py`: Adjust text preprocessing
- `app.py`: Customize UI, thresholds, or visualizations

---

## 📊 Model Performance

### Health Classification
- **Random Forest**: Superior F1-score, feature importance
- **SVM**: Fast predictions, efficient memory usage
- **Features**: 5000+ TF-IDF features + 6 structured features

### Cuisine Discovery (LDA)
- **Topics**: 8-12 discovered cuisine styles
- **Method**: Latent Dirichlet Allocation
- **Optimization**: Perplexity-based topic selection

### Recipe Clustering (KMeans)
- **Clusters**: 8-12 distinct recipe categories
- **Method**: KMeans with cosine similarity
- **Optimization**: Silhouette score maximization

---

## 🛠️ Troubleshooting

### Issue: Models not found
**Solution**: Ensure you've run both training notebooks completely

### Issue: NLTK data not found
**Solution**: The app auto-downloads NLTK data, but you can manually download:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Issue: Port already in use
**Solution**: 
```powershell
streamlit run app.py --server.port 8502
```

### Issue: Memory errors with large models
**Solution**: The training notebooks use sampling for efficiency. Adjust `sample_size` parameters if needed.

---

## 🎨 Features Breakdown

### Interactive Visualizations
- **Gauge Charts**: Health confidence visualization
- **Bar Charts**: Topic distributions, feature importance
- **Color-coded Results**: Intuitive health category colors

### Smart Preprocessing
- **Text Cleaning**: Lowercase, punctuation removal, lemmatization
- **Stopword Filtering**: Custom cooking-specific stopwords
- **Feature Engineering**: TF-IDF, cooking methods, ingredient counts

### User Experience
- **Example Recipes**: Quick-start with pre-loaded examples
- **Responsive Design**: Works on desktop and tablet
- **Real-time Analysis**: Fast predictions (<1 second)
- **Informative UI**: Explanations and tooltips throughout

---

## 🧪 Technical Details

### NLP Pipeline
1. **Tokenization**: Word-level tokenization
2. **Normalization**: Lowercase conversion
3. **Cleaning**: Remove punctuation, numbers, special characters
4. **Stopword Removal**: English + custom cooking terms
5. **Lemmatization**: WordNet lemmatizer
6. **Vectorization**: TF-IDF / Count Vectorizer

### ML Models
- **Random Forest**: 100 trees, max depth 20
- **LinearSVC**: L2 regularization, dual=False
- **LDA**: Online learning, batch size 256
- **KMeans**: 10 clusters, cosine similarity

### Feature Engineering
- **Text Features**: 5000 TF-IDF features with bigrams
- **Structured Features**: 
  - Ingredient count
  - Instruction length
  - Cooking method indicators (baked, fried, grilled, steamed)

---

## 📝 Development Notes

### Extending the System

#### Add New Health Indicators
Edit `vectorizers/preprocessing_tools.pkl` or modify:
```python
healthy_indicators = ['vegetable', 'fruit', ...]
unhealthy_indicators = ['sugar', 'butter', ...]
```

#### Retrain Models
Run the Jupyter notebooks with updated data or parameters.

#### Add New Visualizations
Modify `app.py` visualization functions:
- `create_health_gauge()`
- `create_topic_distribution_chart()`
- `create_feature_importance_chart()`


---

- **Dataset**: RecipeNLG Dataset
- **Libraries**: scikit-learn, NLTK, Streamlit, Plotly
- **Inspiration**: Food science and nutritional research

---

## 🚀 Quick Start Command Summary

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train models (run notebooks)
# - cuisineDiscovery.ipynb (all cells)
# - healthyPrediction.ipynb (all cells)

# Launch app
streamlit run app.py
```

**🎉 Enjoy analyzing recipes with AI!**
