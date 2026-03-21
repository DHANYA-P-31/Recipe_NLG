"""
Smart Recipe Intelligence System
A comprehensive ML-powered recipe analysis platform

Integrates:
- Health Classification (Random Forest & SVM)
- Cuisine Discovery (LDA Topic Modeling)
- Recipe Clustering (KMeans)
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import plotly.express as px
from utils.preprocessing import (
    clean_text, extract_structured_features, 
    preprocess_text_cuisine, create_cuisine_stopwords,
    combine_features_for_health, calculate_health_score
)
from utils.rl_cooking import CookingOptimizationRL
import warnings
warnings.filterwarnings('ignore')


RL_RECIPE_PROFILES = {
    "A": {
        "name": "Balanced Comfort Bowl",
        "description": "Balanced macros with moderate spice and medium prep time.",
    },
    "B": {
        "name": "Bold Spiced Fusion",
        "description": "Spice-forward profile with herbs and longer prep time.",
    },
    "C": {
        "name": "Light Fresh Plate",
        "description": "Lighter, quicker option with mild spice and strong nutrition.",
    },
}

RL_ACTION_DISPLAY_NAMES = {
    0: "Recommend Balanced Comfort Bowl",
    1: "Recommend Bold Spiced Fusion",
    2: "Recommend Light Fresh Plate",
    3: "Suggest ingredient substitution",
    4: "Adjust cooking steps for faster preparation",
}


def get_action_display_name(action_id):
    """Map RL action IDs to user-friendly labels for the UI."""
    return RL_ACTION_DISPLAY_NAMES.get(action_id, f"Action {action_id}")

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK datasets"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_all_models():
    """Load all trained models and vectorizers"""
    models = {}
    
    try:
        # Cuisine Discovery Models (LDA)
        models['lda_model'] = joblib.load('models/cuisine_discovery/lda_model.pkl')
        models['count_vectorizer'] = joblib.load('models/cuisine_discovery/count_vectorizer.pkl')
        models['lda_labels'] = joblib.load('models/cuisine_discovery/lda_labels.pkl')
        
        # Cuisine Clustering Models (KMeans)
        models['kmeans_model'] = joblib.load('models/cuisine_discovery/kmeans_model.pkl')
        models['tfidf_vectorizer'] = joblib.load('models/cuisine_discovery/tfidf_vectorizer.pkl')
        models['kmeans_labels'] = joblib.load('models/cuisine_discovery/kmeans_labels_text.pkl')
        models['cluster_top_words'] = joblib.load('models/cuisine_discovery/cluster_top_words.pkl')
        
        # Health Prediction Models
        models['health_rf_model'] = joblib.load('models/health_prediction/health_rf_model.pkl')
        models['health_svm_model'] = joblib.load('models/health_prediction/health_svm_model.pkl')
        models['health_tfidf_vectorizer'] = joblib.load('models/health_prediction/health_tfidf_vectorizer.pkl')
        models['preprocessing_tools'] = joblib.load('models/health_prediction/preprocessing_tools.pkl')
        
        # Reconstruct feature names from vectorizer (more robust than loading pickle)
        tfidf_feature_names = models['health_tfidf_vectorizer'].get_feature_names_out().tolist()
        structured_feature_names = ['num_ingredients', 'instruction_length', 
                                   'is_baked', 'is_fried', 'is_grilled', 'is_steamed']
        all_feature_names = tfidf_feature_names + structured_feature_names
        
        models['structured_features_info'] = {
            'feature_names': structured_feature_names,
            'all_feature_names': all_feature_names
        }
        
        # Create feature importance dataframe from RF model directly (avoid pickle compatibility issues)
        feature_importances = models['health_rf_model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        models['feature_importance_df'] = feature_importance_df
        
        return models
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("📝 Please run the training notebooks first to generate model files.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()


@st.cache_resource
def load_or_train_rl_agent():
    """Load a trained RL model, or train and persist it if not found."""
    rl_model = CookingOptimizationRL(seed=42)
    model_path = Path('models/cooking_optimization/q_learning_agent.pkl')

    loaded = rl_model.load(model_path)
    if not loaded:
        rl_model.train(episodes=6000)
        rl_model.save(model_path)

    return rl_model


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_health(recipe_title, ingredients, directions, models, model_choice='Random Forest'):
    """
    Predict health category for a recipe
    
    Args:
        recipe_title: Recipe title string
        ingredients: Ingredients text
        directions: Cooking directions (optional)
        models: Dictionary of loaded models
        model_choice: 'Random Forest' or 'SVM'
    
    Returns:
        Dictionary with prediction results
    """
    # Get preprocessing tools
    stop_words = models['preprocessing_tools']['stop_words']
    lemmatizer = WordNetLemmatizer()
    
    # Combine and clean text
    ingredients_text = clean_text(ingredients, stop_words, lemmatizer)
    directions_text = clean_text(directions if directions else "", stop_words, lemmatizer)
    combined_text = ingredients_text + ' ' + directions_text
    
    # Extract TF-IDF features
    tfidf_features = models['health_tfidf_vectorizer'].transform([ingredients_text])
    
    # Extract structured features
    structured_feats = extract_structured_features(combined_text)
    
    # Combine features
    X_combined = combine_features_for_health(tfidf_features, structured_feats)
    
    # Select model
    if model_choice == 'Random Forest':
        model = models['health_rf_model']
        can_get_proba = True
    else:
        model = models['health_svm_model']
        can_get_proba = False
    
    # Make prediction
    prediction = model.predict(X_combined)[0]
    
    # Get confidence scores
    if can_get_proba:
        proba = model.predict_proba(X_combined)[0]
        confidence_dict = {
            label: prob for label, prob in zip(model.classes_, proba)
        }
        confidence = max(proba) * 100
    else:
        # SVM doesn't have predict_proba by default
        confidence_dict = {prediction: 1.0}
        confidence = 100.0
    
    # Calculate rule-based health score
    healthy_indicators = models['preprocessing_tools']['healthy_indicators']
    unhealthy_indicators = models['preprocessing_tools']['unhealthy_indicators']
    health_score = calculate_health_score(combined_text, healthy_indicators, unhealthy_indicators)
    
    # Get top influencing features (Random Forest only)
    top_features = []
    if model_choice == 'Random Forest':
        feature_importance_df = models['feature_importance_df']
        top_20 = feature_importance_df.head(20)
        
        # Check which top features are present in the input
        for _, row in top_20.iterrows():
            if row['feature'] in combined_text.lower():
                top_features.append({
                    'feature': row['feature'],
                    'importance': row['importance']
                })
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'confidence_dict': confidence_dict,
        'health_score': health_score,
        'top_features': top_features[:10],
        'structured_features': structured_feats
    }


def predict_cuisine(recipe_title, ingredients, models):
    """
    Discover cuisine style using LDA topic modeling
    
    Args:
        recipe_title: Recipe title string
        ingredients: Ingredients text
        models: Dictionary of loaded models
    
    Returns:
        Dictionary with cuisine prediction results
    """
    # Create stopwords for cuisine analysis
    stopwords = create_cuisine_stopwords()
    
    # Combine and preprocess text
    combined_text = f"{recipe_title} {ingredients}"
    cleaned_text = preprocess_text_cuisine(combined_text, stopwords)
    
    # Transform using Count Vectorizer
    dtm = models['count_vectorizer'].transform([cleaned_text])
    
    # Get topic distribution
    topic_dist = models['lda_model'].transform(dtm)[0]
    
    # Get dominant topic
    dominant_topic_idx = topic_dist.argmax()
    dominant_topic_prob = topic_dist[dominant_topic_idx]
    
    # Get topic label
    cuisine_style = models['lda_labels'][dominant_topic_idx]
    
    # Get top words for the topic
    vocab = np.array(models['count_vectorizer'].get_feature_names_out())
    topic_words = models['lda_model'].components_[dominant_topic_idx]
    top_indices = topic_words.argsort()[::-1][:10]
    top_keywords = [vocab[i] for i in top_indices]
    
    # Create topic distribution dictionary
    topic_dist_dict = {
        models['lda_labels'][i]: prob 
        for i, prob in enumerate(topic_dist)
    }
    
    return {
        'cuisine_style': cuisine_style,
        'confidence': dominant_topic_prob * 100,
        'topic_distribution': topic_dist_dict,
        'top_keywords': top_keywords,
        'topic_index': dominant_topic_idx
    }


def predict_cluster(recipe_title, ingredients, models):
    """
    Predict recipe cluster using KMeans
    
    Args:
        recipe_title: Recipe title string
        ingredients: Ingredients text
        models: Dictionary of loaded models
    
    Returns:
        Dictionary with cluster prediction results
    """
    # Create stopwords for clustering
    stopwords = create_cuisine_stopwords()
    
    # Combine and preprocess text
    combined_text = f"{recipe_title} {ingredients}"
    cleaned_text = preprocess_text_cuisine(combined_text, stopwords)
    
    # Transform using TF-IDF
    tfidf_features = models['tfidf_vectorizer'].transform([cleaned_text])
    
    # Normalize features
    from sklearn.preprocessing import normalize
    tfidf_features = normalize(tfidf_features)
    
    # Predict cluster
    cluster_id = models['kmeans_model'].predict(tfidf_features)[0]
    
    # Get cluster label and description
    cluster_label = models['kmeans_labels'][cluster_id]
    cluster_keywords = models['cluster_top_words'].get(cluster_id, [])[:10]
    
    # Get distance to centroid (as confidence proxy)
    distances = models['kmeans_model'].transform(tfidf_features)[0]
    confidence = 1 / (1 + distances[cluster_id])  # Convert distance to confidence-like score
    
    return {
        'cluster_id': cluster_id,
        'cluster_label': cluster_label,
        'confidence': confidence * 100,
        'cluster_keywords': cluster_keywords,
        'all_distances': distances
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_health_gauge(confidence, prediction):
    """Create a gauge chart for health prediction confidence"""
    # Set color based on prediction
    color_map = {
        'Healthy': '#28a745',
        'Moderately Healthy': '#ffc107',
        'Unhealthy': '#dc3545'
    }
    color = color_map.get(prediction, '#6c757d')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#f8f9fa'},
                {'range': [50, 75], 'color': '#e9ecef'},
                {'range': [75, 100], 'color': '#dee2e6'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_topic_distribution_chart(topic_distribution):
    """Create bar chart for LDA topic distribution"""
    # Sort by probability
    sorted_topics = sorted(topic_distribution.items(), key=lambda x: x[1], reverse=True)
    cuisines = [x[0] for x in sorted_topics[:8]]
    probabilities = [x[1] * 100 for x in sorted_topics[:8]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities,
            y=cuisines,
            orientation='h',
            marker=dict(
                color=probabilities,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Probability %")
            ),
            text=[f"{p:.1f}%" for p in probabilities],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Cuisine Style Probability Distribution",
        xaxis_title="Probability (%)",
        yaxis_title="Cuisine Style",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_feature_importance_chart(top_features):
    """Create bar chart for top influencing features"""
    if not top_features:
        return None
    
    features = [f['feature'] for f in top_features]
    importances = [f['importance'] for f in top_features]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker=dict(color='steelblue'),
            text=[f"{imp:.4f}" for imp in importances],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Top Influencing Ingredients",
        xaxis_title="Importance Score",
        yaxis_title="Ingredient/Feature",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_confidence_comparison(health_confidence_dict):
    """Create bar chart comparing confidence across health categories"""
    categories = list(health_confidence_dict.keys())
    confidences = [health_confidence_dict[cat] * 100 for cat in categories]
    
    colors = ['#28a745' if cat == 'Healthy' else '#ffc107' if cat == 'Moderately Healthy' else '#dc3545' 
              for cat in categories]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=confidences,
            marker=dict(color=colors),
            text=[f"{c:.1f}%" for c in confidences],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Health Category Probabilities",
        xaxis_title="Health Category",
        yaxis_title="Probability (%)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Smart Recipe Intelligence System",
        page_icon="🍳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #34495e;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background-color: #ecf0f1;
            border-radius: 5px;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            margin: 0.5rem 0;
        }
        .stButton>button {
            width: 100%;
            background-color: #3498db;
            color: white;
            font-size: 1.2rem;
            font-weight: bold;
            padding: 0.75rem;
            border-radius: 8px;
        }
        .rl-panel {
            background: linear-gradient(140deg, #f8fbff 0%, #f6f9fc 100%);
            border: 1px solid #dbe7f4;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .rl-card {
            background: white;
            border: 1px solid #e1e8f0;
            border-radius: 10px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
        }
        .rl-title {
            font-size: 1rem;
            font-weight: 700;
            color: #1f3a5f;
            margin-bottom: 0.25rem;
        }
        .rl-text {
            font-size: 0.92rem;
            color: #46576c;
            margin: 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🍳 Smart Recipe Intelligence System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Recipe Analysis & Discovery Platform</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🔄 Loading ML models..."):
        models = load_all_models()
        rl_model = load_or_train_rl_agent()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.header("⚙️ Configuration")
        
        model_choice = st.selectbox(
            "Health Prediction Model",
            ["Random Forest", "SVM"],
            help="Choose the model for health classification"
        )
        
        st.markdown("---")
        st.header("📚 About")
        st.markdown("""
        This system uses machine learning to analyze recipes:
        
        **🏥 Health Classification**
        - Random Forest / SVM
        - Multi-class prediction
        - Feature importance analysis
        
        **🌎 Cuisine Discovery**
        - LDA Topic Modeling
        - Probabilistic classification
        - Ingredient pattern analysis
        
        **📊 Recipe Clustering**
        - KMeans clustering
        - Recipe similarity detection
        - Style categorization
        """)
        
        st.markdown("---")
        st.info("💡 **Tip:** Provide detailed ingredients for best results!")
        
        # Example recipes button
        if st.button("📝 Load Example Recipe"):
            st.session_state.example_loaded = True
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<p class="section-header">📝 Recipe Input</p>', unsafe_allow_html=True)
        
        # Check if example should be loaded
        if 'example_loaded' in st.session_state and st.session_state.example_loaded:
            default_title = "Grilled Salmon with Steamed Vegetables"
            default_ingredients = """Fresh salmon fillet
Olive oil
Lemon juice
Garlic cloves
Broccoli florets
Carrot slices
Asparagus
Salt and pepper
Fresh herbs (dill, parsley)"""
            default_directions = """Marinate salmon with olive oil, lemon juice, and garlic. 
Grill for 5-7 minutes per side. 
Steam vegetables until tender. 
Season and serve with fresh herbs."""
            st.session_state.example_loaded = False
        else:
            default_title = ""
            default_ingredients = ""
            default_directions = ""
        
        recipe_title = st.text_input(
            "Recipe Title",
            value=default_title,
            placeholder="e.g., Chocolate Chip Cookies"
        )
        
        ingredients = st.text_area(
            "Ingredients",
            value=default_ingredients,
            height=200,
            placeholder="List all ingredients, one per line...\ne.g.,\n2 cups flour\n1 cup sugar\n3 eggs"
        )
        
        directions = st.text_area(
            "Cooking Directions (Optional)",
            value=default_directions,
            height=150,
            placeholder="Describe cooking steps..."
        )
        
        analyze_button = st.button("🔍 Analyze Recipe", type="primary")
    
    with col2:
        st.markdown('<p class="section-header">ℹ️ How It Works</p>', unsafe_allow_html=True)
        
        with st.expander("🏥 Health Classification", expanded=True):
            st.write("""
            **Predicts:** Healthy, Moderately Healthy, or Unhealthy
            
            **Based on:**
            - Ingredient composition
            - Cooking methods
            - Nutritional indicators
            - TF-IDF text features
            """)
        
        with st.expander("🌎 Cuisine Discovery"):
            st.write("""
            **Discovers:** Hidden cuisine patterns
            
            **Using:**
            - LDA Topic Modeling
            - Ingredient co-occurrence
            - Probabilistic classification
            """)
        
        with st.expander("📊 Recipe Clustering"):
            st.write("""
            **Identifies:** Recipe category cluster
            
            **Through:**
            - KMeans clustering
            - TF-IDF similarity
            - Pattern recognition
            """)

        with st.expander("🤖 Cooking Optimization Agent (RL)"):
            st.write("""
            **Recommends:** Best next cooking strategy

            **Optimizes for:**
            - Time feasibility
            - Nutrition quality
            - Ingredient availability
            - User acceptance likelihood

            **Method:** Q-Learning with epsilon-greedy exploration
            """)
    
    # Analysis Results
    if analyze_button:
        if not recipe_title or not ingredients:
            st.error("❌ Please provide at least a recipe title and ingredients!")
        else:
            with st.spinner("🔍 Analyzing your recipe..."):
                # Perform predictions
                health_result = predict_health(
                    recipe_title, ingredients, directions, models, model_choice
                )
                cuisine_result = predict_cuisine(recipe_title, ingredients, models)
                cluster_result = predict_cluster(recipe_title, ingredients, models)
            
            st.success("✅ Analysis Complete!")
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["🏥 Health Analysis", "🌎 Cuisine Discovery", "📊 Cluster Insights"])
            
            # ============================================================
            # TAB 1: HEALTH ANALYSIS
            # ============================================================
            with tab1:
                st.markdown('<p class="section-header">Health Classification Results</p>', unsafe_allow_html=True)
                
                col_a, col_b, col_c = st.columns([2, 2, 3])
                
                with col_a:
                    # Prediction result
                    prediction = health_result['prediction']
                    emoji_map = {
                        'Healthy': '✅',
                        'Moderately Healthy': '⚠️',
                        'Unhealthy': '❌'
                    }
                    st.metric(
                        "Health Category",
                        f"{emoji_map.get(prediction, '📊')} {prediction}"
                    )
                    st.metric(
                        "Model Used",
                        model_choice
                    )
                
                with col_b:
                    st.metric(
                        "Prediction Confidence",
                        f"{health_result['confidence']:.1f}%"
                    )
                    st.metric(
                        "Rule-Based Score",
                        health_result['health_score']
                    )
                
                with col_c:
                    # Gauge chart
                    gauge_fig = create_health_gauge(
                        health_result['confidence'],
                        prediction
                    )
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Category probabilities (if available)
                if len(health_result['confidence_dict']) > 1:
                    st.markdown("#### Probability Distribution")
                    confidence_fig = create_confidence_comparison(
                        health_result['confidence_dict']
                    )
                    st.plotly_chart(confidence_fig, use_container_width=True)
                
                # Structured features
                st.markdown("#### Recipe Characteristics")
                feat_cols = st.columns(3)
                structured_feats = health_result['structured_features']
                
                with feat_cols[0]:
                    st.metric("Ingredient Count", structured_feats['num_ingredients'])
                    st.metric("Is Baked", "Yes" if structured_feats['is_baked'] else "No")
                
                with feat_cols[1]:
                    st.metric("Instruction Length", structured_feats['instruction_length'])
                    st.metric("Is Fried", "Yes" if structured_feats['is_fried'] else "No")
                
                with feat_cols[2]:
                    st.metric("Is Grilled", "Yes" if structured_feats['is_grilled'] else "No")
                    st.metric("Is Steamed", "Yes" if structured_feats['is_steamed'] else "No")
                
                # Top influencing features
                if model_choice == 'Random Forest' and health_result['top_features']:
                    st.markdown("#### Top Influencing Ingredients Found")
                    feature_fig = create_feature_importance_chart(
                        health_result['top_features']
                    )
                    if feature_fig:
                        st.plotly_chart(feature_fig, use_container_width=True)
                    
                    # Display as table
                    feature_df = pd.DataFrame(health_result['top_features'])
                    st.dataframe(
                        feature_df.style.format({'importance': '{:.4f}'}),
                        use_container_width=True
                    )
            
            # ============================================================
            # TAB 2: CUISINE DISCOVERY
            # ============================================================
            with tab2:
                st.markdown('<p class="section-header">Cuisine Discovery Results</p>', unsafe_allow_html=True)
                
                col_d, col_e = st.columns([1, 2])
                
                with col_d:
                    st.metric(
                        "Discovered Cuisine",
                        f"🌍 {cuisine_result['cuisine_style']}"
                    )
                    st.metric(
                        "Confidence",
                        f"{cuisine_result['confidence']:.1f}%"
                    )
                    st.metric(
                        "Topic Index",
                        cuisine_result['topic_index']
                    )
                
                with col_e:
                    st.markdown("#### Top Characteristic Keywords")
                    keywords = cuisine_result['top_keywords']
                    keyword_html = " ".join([
                        f'<span style="background-color:#3498db;color:white;padding:5px 10px;margin:3px;border-radius:5px;display:inline-block;">{kw}</span>'
                        for kw in keywords
                    ])
                    st.markdown(keyword_html, unsafe_allow_html=True)
                
                # Topic distribution chart
                st.markdown("#### All Cuisine Style Probabilities")
                topic_fig = create_topic_distribution_chart(
                    cuisine_result['topic_distribution']
                )
                st.plotly_chart(topic_fig, use_container_width=True)
                
                # Interpretation
                st.info(f"""
                **Interpretation:** Your recipe has a {cuisine_result['confidence']:.1f}% match with 
                **{cuisine_result['cuisine_style']}** style based on ingredient patterns and combinations.
                The characteristic keywords indicate the flavor profile and cooking style.
                """)
            
            # ============================================================
            # TAB 3: CLUSTER INSIGHTS
            # ============================================================
            with tab3:
                st.markdown('<p class="section-header">Recipe Cluster Analysis</p>', unsafe_allow_html=True)
                
                col_f, col_g = st.columns([1, 2])
                
                with col_f:
                    st.metric(
                        "Cluster Category",
                        f"📂 {cluster_result['cluster_label']}"
                    )
                    st.metric(
                        "Cluster ID",
                        cluster_result['cluster_id']
                    )
                    st.metric(
                        "Similarity Score",
                        f"{cluster_result['confidence']:.1f}%"
                    )
                
                with col_g:
                    st.markdown("#### Cluster Keywords")
                    cluster_keywords = cluster_result['cluster_keywords']
                    keyword_html = " ".join([
                        f'<span style="background-color:#e74c3c;color:white;padding:5px 10px;margin:3px;border-radius:5px;display:inline-block;">{kw}</span>'
                        for kw in cluster_keywords
                    ])
                    st.markdown(keyword_html, unsafe_allow_html=True)
                
                # Cluster description
                st.markdown("#### About This Cluster")
                st.success(f"""
                Your recipe belongs to the **{cluster_result['cluster_label']}** category.
                
                Recipes in this cluster typically share similar:
                - Ingredient combinations
                - Cooking techniques
                - Flavor profiles
                
                The similarity score of {cluster_result['confidence']:.1f}% indicates how well 
                your recipe matches the cluster centroid.
                """)
                
                # Distance to all clusters (optional detailed view)
                with st.expander("📊 View Distance to All Clusters"):
                    distances_df = pd.DataFrame({
                        'Cluster': [models['kmeans_labels'][i] for i in range(len(cluster_result['all_distances']))],
                        'Distance': cluster_result['all_distances']
                    }).sort_values('Distance')
                    
                    st.dataframe(distances_df, use_container_width=True)
                    
                    st.caption("Lower distance = Higher similarity")

    # ============================================================
    # RL INTERACTIVE RECOMMENDATION PANEL
    # ============================================================
    st.markdown('<p class="section-header">🤖 Cooking Optimization Agent (Interactive RL)</p>', unsafe_allow_html=True)
    st.markdown('<div class="rl-panel">', unsafe_allow_html=True)
    st.caption("Provide constraints to get a policy-based recommendation and expected reward per action.")
    st.caption("Dataset mode scans all recipes in data/RecipeNLG_dataset.csv and returns top matches.")

    profile_col1, profile_col2, profile_col3 = st.columns(3)
    with profile_col1:
        st.markdown(
            f'''<div class="rl-card"><div class="rl-title">{RL_RECIPE_PROFILES["A"]["name"]}</div><p class="rl-text">{RL_RECIPE_PROFILES["A"]["description"]}</p></div>''',
            unsafe_allow_html=True,
        )
    with profile_col2:
        st.markdown(
            f'''<div class="rl-card"><div class="rl-title">{RL_RECIPE_PROFILES["B"]["name"]}</div><p class="rl-text">{RL_RECIPE_PROFILES["B"]["description"]}</p></div>''',
            unsafe_allow_html=True,
        )
    with profile_col3:
        st.markdown(
            f'''<div class="rl-card"><div class="rl-title">{RL_RECIPE_PROFILES["C"]["name"]}</div><p class="rl-text">{RL_RECIPE_PROFILES["C"]["description"]}</p></div>''',
            unsafe_allow_html=True,
        )

    with st.form("rl_recommendation_form"):
        rl_col1, rl_col2, rl_col3 = st.columns(3)

        with rl_col1:
            available_ingredients = st.multiselect(
                "Available Ingredient Types",
                options=rl_model.env.INGREDIENT_KEYS,
                default=["vegetables", "protein", "grains", "spices", "oil"],
                help="These are abstract ingredient groups used by the RL environment."
            )
            time_available = st.slider("Time Available (minutes)", min_value=10, max_value=90, value=30, step=5)

        with rl_col2:
            diet_pref = st.selectbox("Diet Preference", options=["Non-Vegetarian", "Vegetarian"], index=1)
            spice_pref_label = st.selectbox("Spice Preference", options=["Mild", "Medium", "Hot"], index=1)

        with rl_col3:
            past_recipe_label = st.selectbox(
                "Previous Recipe Profile",
                options=[
                    RL_RECIPE_PROFILES["A"]["name"],
                    RL_RECIPE_PROFILES["B"]["name"],
                    RL_RECIPE_PROFILES["C"]["name"],
                ],
                index=0
            )
            nutrition_a = st.slider(
                f"Estimated Nutrition: {RL_RECIPE_PROFILES['A']['name']}",
                min_value=0.0,
                max_value=1.0,
                value=0.80,
                step=0.05,
            )
            nutrition_b = st.slider(
                f"Estimated Nutrition: {RL_RECIPE_PROFILES['B']['name']}",
                min_value=0.0,
                max_value=1.0,
                value=0.70,
                step=0.05,
            )
            nutrition_c = st.slider(
                f"Estimated Nutrition: {RL_RECIPE_PROFILES['C']['name']}",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05,
            )
            top_n_recipes = st.slider("Number of Dataset Recommendations", min_value=3, max_value=10, value=5, step=1)
            full_scan = st.checkbox(
                "Full Dataset Scan (~4 min, 2.2M recipes)",
                value=False,
                help="Un-checked = quick scan of first 200,000 recipes (~20 sec). Checked = full scan of all 2.2M recipes.",
            )

        recommend_button = st.form_submit_button("🎯 Get RL Recommendation")

    if recommend_button:
        vegetarian_pref = 1 if diet_pref == "Vegetarian" else 0
        spice_pref_map = {"Mild": 0, "Medium": 1, "Hot": 2}
        spice_pref = spice_pref_map[spice_pref_label]
        past_recipe_map = {
            RL_RECIPE_PROFILES["A"]["name"]: 0,
            RL_RECIPE_PROFILES["B"]["name"]: 1,
            RL_RECIPE_PROFILES["C"]["name"]: 2,
        }
        past_recipe_type = past_recipe_map[past_recipe_label]

        state = rl_model.env.build_state_from_constraints(
            available_ingredients=available_ingredients,
            vegetarian_pref=vegetarian_pref,
            spice_pref=spice_pref,
            time_available=time_available,
            past_recipe_type=past_recipe_type,
            nutrition_estimates=[nutrition_a, nutrition_b, nutrition_c],
        )

        recommendation = rl_model.recommend(state)
        q_values = recommendation['q_values']
        display_action = get_action_display_name(recommendation['best_action'])

        st.success(f"Recommended Strategy: {display_action}")

        if recommendation.get('value_source') == 'nearest_neighbors':
            st.info(
                f"Q-values estimated from nearest learned states "
                f"(avg Hamming distance: {recommendation.get('avg_neighbor_distance', 0.0):.2f})."
            )
        elif recommendation.get('value_source') == 'exact':
            st.caption("Q-values retrieved from exact state match in the learned policy.")

        rl_metrics_col1, rl_metrics_col2 = st.columns(2)
        with rl_metrics_col1:
            st.metric("Selected Action ID", recommendation['best_action'])
        with rl_metrics_col2:
            st.metric("Expected Reward (Best)", f"{float(np.max(q_values)):.3f}")

        action_df = pd.DataFrame(
            {
                'action': [get_action_display_name(i) for i in range(len(q_values))],
                'expected_reward': q_values,
            }
        ).sort_values('expected_reward', ascending=False)

        reward_fig = px.bar(
            action_df,
            x='expected_reward',
            y='action',
            orientation='h',
            title='Expected Reward by Action (Q-Values)',
            color='expected_reward',
            color_continuous_scale='Blues'
        )
        reward_fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(reward_fig, use_container_width=True)

        scan_label = "Scanning all 2.2M recipes..." if full_scan else "Quick-scanning first 200,000 recipes..."
        with st.spinner(scan_label):
            ranking_result = rl_model.rank_dataset_recipes(
                dataset_path=Path("data/RecipeNLG_dataset.csv"),
                available_ingredients=available_ingredients,
                vegetarian_pref=vegetarian_pref,
                spice_pref=spice_pref,
                time_available=time_available,
                best_action=recommendation['best_action'],
                nutrition_estimates=[nutrition_a, nutrition_b, nutrition_c],
                top_n=top_n_recipes,
                max_rows=None if full_scan else 200_000,
            )

        if ranking_result.get("warning"):
            st.warning(ranking_result["warning"])
        else:
            st.markdown("#### Top Dataset Recipes")
            st.caption(f"Scanned rows: {ranking_result['scanned_rows']:,}")

            ranked_df = pd.DataFrame(ranking_result["recipes"])
            if not ranked_df.empty:
                ranked_df = ranked_df.rename(
                    columns={
                        "title": "Recipe Title",
                        "score": "Match Score",
                        "estimated_time_min": "Est. Time (min)",
                        "vegetarian_compatible": "Veg Compatible",
                        "source": "Source",
                        "link": "Link",
                    }
                )

                st.dataframe(
                    ranked_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Match Score": st.column_config.NumberColumn(format="%.3f"),
                        "Est. Time (min)": st.column_config.NumberColumn(format="%.1f"),
                        "Link": st.column_config.LinkColumn(display_text="open"),
                    },
                )
            else:
                st.info("No matching recipes found for the current constraints.")

        with st.expander("View Encoded RL State Vector"):
            st.write(state)

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>🧠 Powered by Machine Learning | Built with Streamlit</p>
        <p>Models: Random Forest, SVM, LDA Topic Modeling, KMeans Clustering, Q-Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
