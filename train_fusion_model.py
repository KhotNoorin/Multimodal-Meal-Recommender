import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

# Load embeddings
img_embeddings = np.load("embeddings/image_embeddings.npy")
txt_embeddings = np.load("embeddings/text_embeddings.npy")

# Check consistency
assert img_embeddings.shape[0] == txt_embeddings.shape[0]

# Concatenate image + text
features = np.hstack((img_embeddings, txt_embeddings))  # shape: [N, 2048+384]

# Simulate user preference labels
# 1 = liked, 0 = not liked (for now: randomly assign)
np.random.seed(42)
labels = np.random.choice([0, 1], size=features.shape[0])

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels, test_size=0.2, random_state=42
)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/fusion_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Saved model/fusion_model.pkl and model/scaler.pkl")
