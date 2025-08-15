import numpy as np
import pywt
import matplotlib.pyplot as plt

def compute_wavelet_features(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for c in coeffs:
        energy = np.sum(np.square(c))
        entropy = -np.sum(np.square(c) * np.log(np.square(c) + 1e-10))
        features.append((energy, entropy))
    return features

def plot_wavelet_features(features, label):
    levels = list(range(len(features)))
    energies = [e for e, _ in features]
    entropies = [h for _, h in features]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(levels, energies)
    plt.title(f"{label} - Wavelet Energy")
    plt.xlabel("Level")
    plt.ylabel("Energy")

    plt.subplot(1, 2, 2)
    plt.bar(levels, entropies)
    plt.title(f"{label} - Wavelet Entropy")
    plt.xlabel("Level")
    plt.ylabel("Entropy")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load or regenerate noisy signals
    def generate_signal(freq=50, t_end=1.0, sample_rate=1000):
        t = np.linspace(0, t_end, int(t_end * sample_rate))
        signal = np.sin(2 * np.pi * freq * t)
        return t, signal

    def add_noise(signal, noise_level=0.5):
        noise = noise_level * np.random.randn(len(signal))
        return signal + noise

    t, normal = generate_signal(freq=50)
    _, faulty = generate_signal(freq=50)
    _, harmonic = generate_signal(freq=120)
    faulty += harmonic

    normal_noisy = add_noise(normal)
    faulty_noisy = add_noise(faulty)

    # Compute features
    normal_features = compute_wavelet_features(normal_noisy)
    faulty_features = compute_wavelet_features(faulty_noisy)

    # Print for reference
    print("Normal (noisy) wavelet features:")
    for i, (e, h) in enumerate(normal_features):
        print(f"Level {i}: Energy = {e:.2f}, Entropy = {h:.2f}")

    print("\nFaulty (noisy) wavelet features:")
    for i, (e, h) in enumerate(faulty_features):
        print(f"Level {i}: Energy = {e:.2f}, Entropy = {h:.2f}")

    # Plot
    plot_wavelet_features(normal_features, "Normal")
    plot_wavelet_features(faulty_features, "Faulty")