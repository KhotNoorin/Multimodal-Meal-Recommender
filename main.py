import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load recipe metadata
df = pd.read_csv("data/recipes.csv")
image_embeddings = np.load("embeddings/image_embeddings.npy")
text_embeddings = np.load("embeddings/text_embeddings.npy")

# Concatenate image + text embeddings
all_embeddings = np.hstack((image_embeddings, text_embeddings))

# Load trained model and scaler
model = joblib.load("model/fusion_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Load text embedding model (same one used before)
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

def get_user_input_embedding(user_text):
    """Convert user query to a BERT embedding"""
    text_emb = text_encoder.encode([user_text])  # shape: (1, 384)
    return text_emb

def recommend(user_text, top_k=5):
    # Convert user text to BERT embedding
    user_text_emb = get_user_input_embedding(user_text)

    # Combine with average image embedding (simple simulation)
    avg_img_emb = np.mean(image_embeddings, axis=0, keepdims=True)
    user_full_emb = np.hstack((avg_img_emb, user_text_emb))  # shape: (1, 2432)

    # Scale input using same scaler
    user_full_emb_scaled = scaler.transform(user_full_emb)

    # Predict like/dislike for all recipes
    preds = model.predict_proba(scaler.transform(all_embeddings))[:, 1]  # prob of "liked"

    # Compute similarity between user query and each recipe
    sim = cosine_similarity(user_full_emb_scaled, scaler.transform(all_embeddings)).flatten()

    # Combine: sort by predicted preference × similarity
    combined_score = preds * sim
    top_indices = np.argsort(combined_score)[::-1][:top_k]

    recommendations = df.iloc[top_indices][["name", "ingredients", "calories", "allergens"]]
    print(f"\nTop {top_k} Recommendations based on your query:\n")
    for idx, row in recommendations.iterrows():
        print(f" {row['name']}")
        print(f"   Ingredients: {row['ingredients']}")
        print(f"   Calories: {row['calories']}")
        print(f"   Allergens: {row['allergens']}")
        print("")

if __name__ == "__main__":
    print("Enter your meal preference (text or ingredients):")
    query = input(">> ")
    recommend(query, top_k=5)
