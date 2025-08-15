import numpy as np
import matplotlib.pyplot as plt

fs = 1000
t = np.linspace(0, 1, fs)

signal_clean = np.sin(2 * np.pi * 50 * t)

noise = np.random.normal(0, 0.3, fs)
signal_noisy = signal_clean + noise

fault_signal = signal_noisy.copy()
fault_signal[400:600] += 2

plt.figure(figsize=(12, 6))
plt.plot(t, fault_signal, label='Faulty Signal', color='red')
plt.plot(t, signal_noisy, label='Noisy Signal', alpha=0.5)
plt.plot(t, signal_clean, label='Clean Signal', linestyle='--')
plt.title("Synthetic Motor Signal with Fault")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()