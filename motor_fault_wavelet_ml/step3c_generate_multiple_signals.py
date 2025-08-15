import numpy as np
import os
import matplotlib.pyplot as plt

def generate_signal(freq, duration=1.0, sampling_rate=1000):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal

def add_noise(signal, noise_level=0.5):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

# Parameters
num_samples = 100
sampling_rate = 1000
duration = 1.0

normal_freq = 50
faulty_freqs = [90, 120]

signals = []
labels = []

# Generate normal signals
for _ in range(num_samples):
    _, s = generate_signal(normal_freq, duration, sampling_rate)
    s_noisy = add_noise(s, noise_level=np.random.uniform(0.3, 1.0))
    signals.append(s_noisy)
    labels.append(0)

# Generate faulty signals (multiple fault types)
for freq in faulty_freqs:
    for _ in range(num_samples):
        _, s = generate_signal(freq, duration, sampling_rate)
        s_noisy = add_noise(s, noise_level=np.random.uniform(0.3, 1.0))
        signals.append(s_noisy)
        labels.append(1)

# Convert to array
signals = np.array(signals)
labels = np.array(labels)

# Save data for next step
np.savez("generated_signals.npz", signals=signals, labels=labels)

# Plot sample signals
plt.figure(figsize=(10, 6))
plt.plot(signals[0], label="Normal")
plt.plot(signals[-1], label="Faulty")
plt.title("Sample Noisy Signals")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()