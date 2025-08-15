import numpy as np
import pywt
import matplotlib.pyplot as plt

fs = 1000
t = np.linspace(0, 1, fs)
signal_clean = np.sin(2 * np.pi * 50 * t)
noise = np.random.normal(0, 0.3, fs)
signal_noisy = signal_clean + noise
fault_signal = signal_noisy.copy()
fault_signal[400:600] += 2

wavelet = 'db4'
coeffs = pywt.wavedec(fault_signal, wavelet, level=4)

features = []
for i, coeff in enumerate(coeffs):
    energy = np.sum(np.square(coeff))
    entropy = -np.sum(np.where(coeff != 0, coeff**2 * np.log(np.abs(coeff) + 1e-12), 0))
    features.append((energy, entropy))

features_array = np.array(features)

print("Wavelet Features (energy, entropy) per level:")
for i, (e, h) in enumerate(features):
    print(f"Level {i}: Energy = {e:.2f}, Entropy = {h:.2f}")

plt.figure(figsize=(10, 6))
for i in range(len(coeffs)):
    plt.subplot(len(coeffs), 1, i+1)
    plt.plot(coeffs[i])
    plt.title(f'Wavelet Coefficients - Level {i}')
plt.tight_layout()

plt.savefig("wavelet_plot.png", dpi=300)
plt.show()