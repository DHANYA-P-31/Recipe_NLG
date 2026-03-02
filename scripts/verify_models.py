"""
Model Verification Script
Checks if all required models and vectorizers are present
"""

import os
import sys

def check_file(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def main():
    print("="*70)
    print("Smart Recipe Intelligence System - Model Verification")
    print("="*70)
    print()
    
    all_present = True
    
    # Cuisine Discovery Models (LDA)
    print("📊 Cuisine Discovery Models (LDA):")
    print("-" * 70)
    all_present &= check_file("models/cuisine_discovery/lda_model.pkl", "LDA Model")
    all_present &= check_file("models/cuisine_discovery/count_vectorizer.pkl", "Count Vectorizer")
    all_present &= check_file("models/cuisine_discovery/lda_labels.pkl", "LDA Labels")
    print()
    
    # Cuisine Clustering Models (KMeans)
    print("🔍 Cuisine Clustering Models (KMeans):")
    print("-" * 70)
    all_present &= check_file("models/cuisine_discovery/kmeans_model.pkl", "KMeans Model")
    all_present &= check_file("models/cuisine_discovery/tfidf_vectorizer.pkl", "TF-IDF Vectorizer")
    all_present &= check_file("models/cuisine_discovery/kmeans_labels_text.pkl", "KMeans Labels")
    all_present &= check_file("models/cuisine_discovery/cluster_top_words.pkl", "Cluster Keywords")
    print()
    
    # Health Prediction Models
    print("🏥 Health Prediction Models:")
    print("-" * 70)
    all_present &= check_file("models/health_prediction/health_rf_model.pkl", "Random Forest Model")
    all_present &= check_file("models/health_prediction/health_svm_model.pkl", "SVM Model")
    all_present &= check_file("models/health_prediction/health_tfidf_vectorizer.pkl", "Health TF-IDF Vectorizer")
    all_present &= check_file("models/health_prediction/preprocessing_tools.pkl", "Preprocessing Tools")
    print("   ℹ️  Note: Feature importance is computed on-the-fly from RF model")
    print()
    
    # Utils
    print("🛠️ Utilities:")
    print("-" * 70)
    all_present &= check_file("app/utils/preprocessing.py", "Preprocessing Module")
    all_present &= check_file("app/app.py", "Main Application")
    print()
    
    print("="*70)
    if all_present:
        print("✅ SUCCESS: All models and files are present!")
        print("="*70)
        print()
        print("You can now launch the application:")
        print("  streamlit run app/app.py")
        print()
        print("Or use the launch scripts:")
        print("  PowerShell: .\\run_app.ps1")
        print("  CMD: run_app.bat")
        return 0
    else:
        print("❌ ERROR: Some models or files are missing!")
        print("="*70)
        print()
        print("NEXT STEPS:")
        print("1. Ensure you've run all cells in notebooks/cuisineDiscovery.ipynb")
        print("2. Ensure you've run all cells in notebooks/healthyPrediction.ipynb")
        print("3. Check that models/ directories exist with all files")
        print()
        print("See docs/SETUP_GUIDE.md for detailed instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
