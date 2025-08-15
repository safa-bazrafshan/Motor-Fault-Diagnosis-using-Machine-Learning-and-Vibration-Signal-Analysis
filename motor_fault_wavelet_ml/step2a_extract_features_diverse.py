# step2a_extract_features_diverse.py
import pandas as pd
import numpy as np
import pywt
import os

# Load dataset
df = pd.read_csv("datasets/motor_signals_diverse.csv")

# Separate features and labels
labels = df["label"]
signals = df.drop("label", axis=1).values

# Function to extract statistical + wavelet features from a signal
def extract_features(signal):
    features = []
    # Statistical features
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.max(signal))
    features.append(np.min(signal))
    features.append(np.median(signal))
    features.append(np.sum(np.abs(signal)))
    features.append(np.sum(signal ** 2))

    # Wavelet features (using 'db4', level 3)
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    for c in coeffs:
        features.append(np.mean(c))
        features.append(np.std(c))
        features.append(np.max(c))
        features.append(np.min(c))

    return features

# Extract features for all signals
feature_list = [extract_features(sig) for sig in signals]

# Create DataFrame
feature_df = pd.DataFrame(feature_list)
feature_df["label"] = labels

# Save dataset
os.makedirs("datasets", exist_ok=True)
output_path = "datasets/motor_features_diverse.csv"
feature_df.to_csv(output_path, index=False)

print(f"Feature dataset saved at: {output_path}")
print(f"Shape: {feature_df.shape}")