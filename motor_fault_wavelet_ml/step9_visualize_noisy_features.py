import numpy as np
import pywt
import matplotlib.pyplot as plt

# Load a sample noisy signal
t = np.linspace(0, 1, 1000)
signal_normal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(len(t))
signal_faulty = (np.sin(2 * np.pi * 50 * t) +
                 np.sin(2 * np.pi * 120 * t) +
                 0.5 * np.random.randn(len(t)))

# Wavelet decomposition
wavelet = 'db4'
coeffs_normal = pywt.wavedec(signal_normal, wavelet, level=5)
coeffs_faulty = pywt.wavedec(signal_faulty, wavelet, level=5)

# Plot original signals
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, signal_normal)
plt.title("Normal Noisy Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(t, signal_faulty)
plt.title("Faulty Noisy Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# Plot wavelet details comparison
plt.figure(figsize=(12, 10))
for i, (cN, cF) in enumerate(zip(coeffs_normal[1:], coeffs_faulty[1:]), 1):
    plt.subplot(5, 2, 2*i-1)
    plt.plot(cN)
    plt.title(f"Normal - Detail Level {i}")

    plt.subplot(5, 2, 2*i)
    plt.plot(cF)
    plt.title(f"Faulty - Detail Level {i}")

plt.tight_layout()
plt.show()