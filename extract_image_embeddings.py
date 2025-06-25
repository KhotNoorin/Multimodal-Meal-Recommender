import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Paths
image_dir = "data/images"
recipe_csv = "data/recipes.csv"
output_path = "embeddings/image_embeddings.npy"
os.makedirs("embeddings", exist_ok=True)

# Load recipe CSV
df = pd.read_csv(recipe_csv)
image_ids = df["id"].tolist()

# Define image preprocessor for ResNet50
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # converts to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenet stats
                         std=[0.229, 0.224, 0.225])
])

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final FC
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Extract embeddings
embeddings = []

for img_id in tqdm(image_ids, desc="Extracting image features"):
    img_path = os.path.join(image_dir, f"{img_id}.jpg")
    
    try:
        image = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]
        
        with torch.no_grad():
            output = model(input_tensor)  # shape: [1, 2048, 1, 1]
            features = output.squeeze().cpu().numpy()  # shape: [2048]
        
        embeddings.append(features)
    
    except Exception as e:
        print(f"Skipping image {img_path}: {e}")
        embeddings.append(np.zeros(2048))  # fallback to zero vector

# Save to .npy file
embeddings = np.vstack(embeddings)  # shape: [N, 2048]
np.save(output_path, embeddings)

print(f"Saved {embeddings.shape[0]} image embeddings to {output_path}")
