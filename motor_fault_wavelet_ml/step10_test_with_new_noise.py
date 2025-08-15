# step10_test_with_new_noise.py
import numpy as np
import pywt
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to generate synthetic signal
def generate_signal(freq=50, length=1.0, fs=1000, noise_level=0.5):
    t = np.linspace(0, length, int(fs * length), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    noisy_signal = signal + np.random.normal(0, noise_level, size=signal.shape)
    return t, noisy_signal

# Function to extract wavelet features
def extract_wavelet_features(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for coeff in coeffs:
        energy = np.sum(np.square(coeff))
        entropy = -np.sum(np.square(coeff) * np.log(np.square(coeff) + 1e-12))
        features.extend([energy, entropy])
    return features

# Load trained models and scaler
rf_model = joblib.load("random_forest_noisy.pkl")
svm_model = joblib.load("svm_noisy.pkl")
xgb_model = joblib.load("xgboost_noisy.pkl")
scaler = joblib.load("scaler_noisy.pkl")

# Generate a new signal with different noise
t, test_signal = generate_signal(freq=50, noise_level=1.0)  # Higher noise

# Extract features
features = extract_wavelet_features(test_signal)
features_scaled = scaler.transform([features])

# Predict with each model
pred_rf = rf_model.predict(features_scaled)[0]
pred_svm = svm_model.predict(features_scaled)[0]
pred_xgb = xgb_model.predict(features_scaled)[0]

# Map label to class name
label_map = {0: "Normal", 1: "Faulty"}

print(f"RandomForest Prediction: {label_map[pred_rf]}")
print(f"SVM Prediction: {label_map[pred_svm]}")
print(f"XGBoost Prediction: {label_map[pred_xgb]}")

# Plot the test signal
plt.figure(figsize=(10, 4))
plt.plot(t, test_signal, label="Test Signal (High Noise)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Test Signal with Higher Noise")
plt.legend()
plt.tight_layout()
plt.show()