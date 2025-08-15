import numpy as np
import pywt
import pandas as pd

def extract_wavelet_features(signal, wavelet="db4", level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = {}
    for i, c in enumerate(coeffs):
        energy = np.sum(np.square(c))
        entropy = -np.sum(np.square(c) * np.log(np.square(c) + 1e-12))
        features[f"energy_L{i}"] = energy
        features[f"entropy_L{i}"] = entropy
    return features

# Load signals
data = np.load("noisy_signals.npy", allow_pickle=True).item()

rows = []
for label, signals in data.items():
    for sig in signals:
        feats = extract_wavelet_features(sig)
        feats["label"] = 0 if label == "normal" else 1
        rows.append(feats)

# Create dataframe
df = pd.DataFrame(rows)
df.to_csv("motor_wavelet_dataset_multiple_noisy.csv", index=False)

print(f"Dataset saved as motor_wavelet_dataset_multiple_noisy.csv with shape {df.shape}")