"""
Improve LDA cuisine labels
Run this to create better, more descriptive labels
"""

import joblib
import numpy as np

# Load the LDA model
lda_model = joblib.load('models/cuisine_discovery/lda_model.pkl')
count_vectorizer = joblib.load('models/cuisine_discovery/count_vectorizer.pkl')

# Get vocabulary
vocab = np.array(count_vectorizer.get_feature_names_out())

print("=" * 80)
print("ANALYZING LDA TOPICS FOR BETTER LABELS")
print("=" * 80)

# Analyze each topic
for topic_idx in range(lda_model.n_components):
    print(f"\nTopic {topic_idx}:")
    topic_words = lda_model.components_[topic_idx]
    top_indices = topic_words.argsort()[::-1][:20]
    top_words = [vocab[i] for i in top_indices]
    print("Top words:", ", ".join(top_words))
    print("-" * 80)

# Create improved labels based on analysis
improved_labels = [
    "Savory Main Dishes",             # Topic 0: pepper, garlic, onion, chicken, beef
    "Fruit Desserts & Pies",          # Topic 1: juice, lemon, orange, pineapple, cream, pie
    "Baking & Sweet Treats",          # Topic 2: vanilla, baking, chocolate, eggs, cake
    "Creamy Casseroles",              # Topic 3: cheese, cream, chicken, soup, casserole
    "Mediterranean & Salads",         # Topic 4: olive, fresh, lemon, pepper, salad
    "Breads & Pizza",                 # Topic 5: bread, pizza, yeast, rolls
    "Healthy Light Meals",            # Topic 6: low-fat, shrimp, yogurt, almonds, sesame
    "Spiced & Aromatic Dishes"        # Topic 7: cinnamon, ginger, nutmeg, sweet spices
]

print("\n" + "=" * 80)
print("PROPOSED IMPROVED LABELS:")
print("=" * 80)
for i, label in enumerate(improved_labels):
    print(f"Topic {i}: {label}")

# Save improved labels
joblib.dump(improved_labels, 'models/cuisine_discovery/lda_labels_improved.pkl')
print("\n✅ Saved improved labels to: models/cuisine_discovery/lda_labels_improved.pkl")
print("\nTo use these labels, replace lda_labels.pkl with lda_labels_improved.pkl")
