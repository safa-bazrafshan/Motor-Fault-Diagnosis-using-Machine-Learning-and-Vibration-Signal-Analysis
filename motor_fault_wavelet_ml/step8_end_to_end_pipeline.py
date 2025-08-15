import numpy as np
import pandas as pd
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Generate noisy signals
def generate_signal(freq=50, noise_level=0.2, length=1.0, fs=1000):
    t = np.linspace(0, length, int(fs * length), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    noise = noise_level * np.random.randn(len(t))
    return t, signal + noise

# 2. Extract wavelet features
def wavelet_features(signal, wavelet="db4", level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for c in coeffs:
        energy = np.sum(np.square(c))
        entropy = -np.sum(c ** 2 * np.log(np.abs(c) + 1e-12))
        features.extend([energy, entropy])
    return features

# 3. Create dataset
def create_dataset(n_samples=50):
    X, y = [], []
    for _ in range(n_samples):
        # Normal signal
        _, sig_n = generate_signal(freq=50)
        X.append(wavelet_features(sig_n))
        y.append(0)

        # Faulty signal
    _, sig_base = generate_signal(freq=50)
    _, sig_fault = generate_signal(freq=120)
    sig_f = sig_base + sig_fault
    X.append(wavelet_features(sig_f))
    y.append(1)
    return np.array(X), np.array(y)

# 4. Train model
X, y = create_dataset(n_samples=50)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "motor_fault_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved.")

# 5. Test prediction
_, test_sig = generate_signal(freq=50)  # Normal example
test_features = wavelet_features(test_sig)
test_features_scaled = scaler.transform([test_features])
pred = model.predict(test_features_scaled)[0]

print("Predicted class:", "Normal" if pred == 0 else "Faulty")