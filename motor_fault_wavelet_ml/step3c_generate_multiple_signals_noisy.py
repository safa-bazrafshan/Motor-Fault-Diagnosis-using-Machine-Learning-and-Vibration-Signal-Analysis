import numpy as np

def generate_signal(freq=50, sampling_rate=1000, duration=1, noise_level=0.2):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return t, signal + noise

# Parameters
num_samples_per_class = 50
signals = {"normal": [], "faulty": []}

# Generate normal signals (one frequency + noise)
for _ in range(num_samples_per_class):
    _, sig = generate_signal(freq=50, noise_level=0.3)
    signals["normal"].append(sig)

# Generate faulty signals (two frequencies + noise)
for _ in range(num_samples_per_class):
    _, sig1 = generate_signal(freq=50, noise_level=0.3)
    _, sig2 = generate_signal(freq=120, noise_level=0.3)
    faulty_signal = sig1 + sig2
    signals["faulty"].append(faulty_signal)

# Save to file
np.save("noisy_signals.npy", signals)
print(f"Generated {num_samples_per_class} normal and {num_samples_per_class} faulty signals with noise.")
print("Saved to noisy_signals.npy")