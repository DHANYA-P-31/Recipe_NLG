"""
Test script to verify model loading
"""

import joblib
import sys

print("Testing model loading...")
print("=" * 70)

try:
    print("\n1. Testing Cuisine Discovery Models (LDA)...")
    lda_model = joblib.load('models/cuisine_discovery/lda_model.pkl')
    count_vectorizer = joblib.load('models/cuisine_discovery/count_vectorizer.pkl')
    lda_labels = joblib.load('models/cuisine_discovery/lda_labels.pkl')
    print("   ✅ LDA models loaded successfully")
    
    print("\n2. Testing Cuisine Clustering Models (KMeans)...")
    kmeans_model = joblib.load('models/cuisine_discovery/kmeans_model.pkl')
    tfidf_vectorizer = joblib.load('models/cuisine_discovery/tfidf_vectorizer.pkl')
    kmeans_labels = joblib.load('models/cuisine_discovery/kmeans_labels_text.pkl')
    cluster_top_words = joblib.load('models/cuisine_discovery/cluster_top_words.pkl')
    print("   ✅ KMeans models loaded successfully")
    
    print("\n3. Testing Health Prediction Models...")
    health_rf_model = joblib.load('models/health_prediction/health_rf_model.pkl')
    health_svm_model = joblib.load('models/health_prediction/health_svm_model.pkl')
    health_tfidf_vectorizer = joblib.load('models/health_prediction/health_tfidf_vectorizer.pkl')
    preprocessing_tools = joblib.load('models/health_prediction/preprocessing_tools.pkl')
    print("   ✅ Health models loaded successfully")
    
    print("\n4. Reconstructing feature names...")
    tfidf_feature_names = health_tfidf_vectorizer.get_feature_names_out().tolist()
    structured_feature_names = ['num_ingredients', 'instruction_length', 
                               'is_baked', 'is_fried', 'is_grilled', 'is_steamed']
    all_feature_names = tfidf_feature_names + structured_feature_names
    print(f"   ✅ Feature names reconstructed: {len(all_feature_names)} total features")
    
    print("\n5. Creating feature importance dataframe...")
    import pandas as pd
    feature_importances = health_rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    print(f"   ✅ Feature importance dataframe created: {len(feature_importance_df)} features")
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS: All models loaded without errors!")
    print("=" * 70)
    
    print("\nModel Summary:")
    print(f"  - LDA Topics: {lda_model.n_components}")
    print(f"  - KMeans Clusters: {kmeans_model.n_clusters}")
    print(f"  - Random Forest Trees: {health_rf_model.n_estimators}")
    print(f"  - TF-IDF Features: {len(tfidf_feature_names)}")
    print(f"  - Structured Features: {len(structured_feature_names)}")
    print(f"  - Total Features: {len(all_feature_names)}")
    
    sys.exit(0)
    
except Exception as e:
    print("\n" + "=" * 70)
    print(f"❌ ERROR: {str(e)}")
    print("=" * 70)
    import traceback
    traceback.print_exc()
    sys.exit(1)
