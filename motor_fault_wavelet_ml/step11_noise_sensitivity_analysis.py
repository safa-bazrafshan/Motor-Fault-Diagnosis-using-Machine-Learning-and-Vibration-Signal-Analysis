# step11_noise_sensitivity_analysis.py
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# Function to generate signal
def generate_signal(freq=50, duration=1.0, sample_rate=1000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal

# Function to add noise
def add_noise(signal, noise_level):
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

# Extract wavelet features
def extract_wavelet_features(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=5)
    features = []
    for c in coeffs:
        energy = np.sum(np.square(c))
        entropy = -np.sum((c ** 2) * np.log(np.abs(c) + 1e-12))
        features.extend([energy, entropy])
    return features

# Load trained model and scaler
model = joblib.load("random_forest_noisy.pkl")
scaler = joblib.load("scaler_noisy.pkl")

noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
accuracies = []

# Generate test data for each noise level
for nl in noise_levels:
    X_test = []
    y_test = []
    for _ in range(50):
        _, normal_signal = generate_signal(freq=50)
        _, sig1 = generate_signal(freq=50)
        _, sig2 = generate_signal(freq=120)
        faulty_signal = sig1 + sig2

        normal_signal = add_noise(normal_signal, nl)
        faulty_signal = add_noise(faulty_signal, nl)

        X_test.append(extract_wavelet_features(normal_signal))
        y_test.append(0)

        X_test.append(extract_wavelet_features(faulty_signal))
        y_test.append(1)

    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Plot results
plt.figure(figsize=(6, 4))
plt.plot(noise_levels, accuracies, marker='o')
plt.xlabel("Noise Level (std dev)")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs Noise Level")
plt.grid(True)
plt.tight_layout()
plt.show()