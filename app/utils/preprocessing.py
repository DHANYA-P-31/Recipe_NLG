"""
Preprocessing utilities for Smart Recipe Intelligence System
Handles text cleaning and feature extraction for recipe analysis
"""

import re
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix


def clean_text(text, stop_words, lemmatizer):
    """
    Clean and preprocess text data for recipe analysis
    
    Args:
        text: Raw text string (ingredients or directions)
        stop_words: Set of stopwords to remove
        lemmatizer: NLTK WordNetLemmatizer instance
    
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and punctuation, keep only alphabets and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words 
             if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)


def extract_structured_features(text):
    """
    Extract structured features from recipe text
    
    Args:
        text: Cleaned recipe text
    
    Returns:
        Dictionary of structured features
    """
    text_lower = str(text).lower()
    
    features = {
        'num_ingredients': len(text.split()),
        'instruction_length': len(text.split()),
        'is_baked': 1 if 'bake' in text_lower else 0,
        'is_fried': 1 if 'fry' in text_lower or 'fried' in text_lower else 0,
        'is_grilled': 1 if 'grill' in text_lower else 0,
        'is_steamed': 1 if 'steam' in text_lower else 0
    }
    
    return features


def preprocess_text_cuisine(text, stopwords):
    """
    Preprocess text for cuisine discovery (LDA/KMeans)
    Follows the same preprocessing as training
    
    Args:
        text: Raw text string
        stopwords: Set of stopwords including cooking-specific terms
    
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation and numbers
    punct_num_regex = re.compile(r"[^a-z\s]")
    text = punct_num_regex.sub(" ", text)
    
    # Remove stopwords
    tokens = [t for t in text.split() if t not in stopwords and len(t) > 2]
    
    return " ".join(tokens)


def create_cuisine_stopwords():
    """
    Create custom stopwords for cuisine discovery
    
    Returns:
        Set of stopwords
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    cooking_stopwords = {
        "salt", "water", "oil", "sugar", "butter", "flour",
        "cup", "cups", "tsp", "tbsp", "teaspoon", "tablespoon",
        "teaspoons", "tablespoons", "oz", "ounce", "ounces", "lb", "lbs",
        "pkg", "package", "can", "cans", "soda", "mix", "instant"
    }
    
    return set(ENGLISH_STOP_WORDS).union(cooking_stopwords)


def combine_features_for_health(tfidf_features, structured_features):
    """
    Combine TF-IDF features with structured features
    
    Args:
        tfidf_features: Sparse matrix from TF-IDF vectorizer
        structured_features: Dictionary or array of structured features
    
    Returns:
        Combined sparse matrix
    """
    if isinstance(structured_features, dict):
        structured_array = np.array([[
            structured_features['num_ingredients'],
            structured_features['instruction_length'],
            structured_features['is_baked'],
            structured_features['is_fried'],
            structured_features['is_grilled'],
            structured_features['is_steamed']
        ]])
    else:
        structured_array = structured_features
    
    return hstack([tfidf_features, csr_matrix(structured_array)])


def calculate_health_score(text, healthy_indicators, unhealthy_indicators):
    """
    Calculate health score based on ingredient indicators
    
    Args:
        text: Recipe text
        healthy_indicators: List of healthy ingredient keywords
        unhealthy_indicators: List of unhealthy ingredient keywords
    
    Returns:
        Health score (integer)
    """
    if pd.isna(text):
        return 0
    
    text = str(text).lower()
    score = 0
    
    # Count healthy indicators (+1 each)
    for indicator in healthy_indicators:
        if indicator in text:
            score += 1
    
    # Count unhealthy indicators (-1 each)
    for indicator in unhealthy_indicators:
        if indicator in text:
            score -= 1
    
    return score


def get_top_features_for_prediction(feature_importances, feature_names, text, n_top=10):
    """
    Get top influencing features for a given prediction
    
    Args:
        feature_importances: Array of feature importance values
        feature_names: List of feature names
        text: Input text to check feature presence
        n_top: Number of top features to return
    
    Returns:
        List of tuples (feature_name, importance, present_in_text)
    """
    text_lower = str(text).lower()
    
    # Create list of (feature, importance, presence)
    feature_data = []
    for fname, importance in zip(feature_names, feature_importances):
        is_present = fname in text_lower
        feature_data.append((fname, importance, is_present))
    
    # Sort by importance
    feature_data.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N
    return feature_data[:n_top]
