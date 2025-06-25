import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
input_csv = "data/recipes.csv"
output_path = "embeddings/text_embeddings.npy"
os.makedirs("embeddings", exist_ok=True)

# Load ingredients column
df = pd.read_csv(input_csv)
texts = df["ingredients"].astype(str).tolist()  # make sure it's string

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode in batches (automatically handled)
print("Encoding text embeddings...")
text_embeddings = model.encode(texts, show_progress_bar=True)

# Save to file
np.save(output_path, text_embeddings)
print(f"Saved {len(text_embeddings)} text embeddings to {output_path}")
