import numpy as np
import pywt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Generate a noisy signal
def generate_signal(freq=50, sampling_rate=1000, duration=1.0, noise_level=0.2):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return t, signal + noise


# Extract wavelet features
def wavelet_features(signal, wavelet="db4", level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for c in coeffs:
        energy = np.sum(np.square(c))
        entropy = -np.sum(np.square(c) * np.log(np.square(c) + 1e-12))
        features.extend([energy, entropy])
    return features


# Create dataset with noisy signals
def create_dataset(n_samples=50):
    X = []
    y = []
    for _ in range(n_samples):
        # Normal noisy
        _, sig_n = generate_signal(freq=50)
        X.append(wavelet_features(sig_n))
        y.append(0)

        # Faulty noisy
        _, sig_f1 = generate_signal(freq=50)
        _, sig_f2 = generate_signal(freq=120)
        sig_f = sig_f1 + sig_f2
        X.append(wavelet_features(sig_f))
        y.append(1)
    return np.array(X), np.array(y)


# Main pipeline
if __name__ == "__main__":
    X, y = create_dataset(n_samples=50)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy (noisy pipeline): {acc:.2f}")

    # Save model
    joblib.dump(pipeline, "rf_wavelet_noisy_pipeline.pkl")
    print("Noisy pipeline model saved.")

    # Test with a new noisy signal
    _, test_sig = generate_signal(freq=50)
    test_features = np.array(wavelet_features(test_sig)).reshape(1, -1)
    prediction = pipeline.predict(test_features)
    print(f"Predicted class for test noisy signal: {'Faulty' if prediction[0] == 1 else 'Normal'}")