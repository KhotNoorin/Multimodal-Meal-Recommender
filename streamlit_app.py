import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os

# Load everything
df = pd.read_csv("data/recipes.csv")
image_embeddings = np.load("embeddings/image_embeddings.npy")
text_embeddings = np.load("embeddings/text_embeddings.npy")
all_embeddings = np.hstack((image_embeddings, text_embeddings))

model = joblib.load("model/fusion_model.pkl")
scaler = joblib.load("model/scaler.pkl")
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

def get_user_embedding(user_text):
    """Get BERT embedding of input text"""
    return text_encoder.encode([user_text])

def recommend(user_text, top_k=5):
    # Encode user input
    user_text_emb = get_user_embedding(user_text)
    avg_img_emb = np.mean(image_embeddings, axis=0, keepdims=True)
    user_full_emb = np.hstack((avg_img_emb, user_text_emb))

    # Scale
    user_full_emb_scaled = scaler.transform(user_full_emb)

    # Predict preference scores for all
    preds = model.predict_proba(scaler.transform(all_embeddings))[:, 1]
    sim = cosine_similarity(user_full_emb_scaled, scaler.transform(all_embeddings)).flatten()

    # Final score: combine similarity × classifier
    combined = preds * sim
    top_indices = np.argsort(combined)[::-1][:top_k]

    return df.iloc[top_indices], top_indices

# --- Streamlit UI ---
st.set_page_config(page_title="Multimodal Meal Recommender", layout="wide")

st.title("Multimodal Meal Recommender")
st.markdown("Enter what you feel like eating — we'll suggest recipes using image + text intelligence!")

user_input = st.text_input("What do you feel like eating?", placeholder="e.g., something with chicken and garlic, low in calories")

if user_input:
    with st.spinner("Analyzing your preferences..."):
        top_df, indices = recommend(user_input)

    st.success("Here are your top recommendations:")

    for i, row in top_df.iterrows():
        col1, col2 = st.columns([1, 3])
        img_path = f"data/images/{row['id']}.jpg"

        # Show image if it exists
        if os.path.exists(img_path):
            image = Image.open(img_path)
            col1.image(image, caption=row['name'], width=150)
        else:
            col1.text("No Image")

        # Show metadata
        col2.markdown(f"** {row['name']}**")
        col2.markdown(f"**Ingredients:** {row['ingredients']}")
        col2.markdown(f"**Calories:** {row['calories']}")
        col2.markdown(f"**Allergens:** {row['allergens']}")
        col2.markdown("---")