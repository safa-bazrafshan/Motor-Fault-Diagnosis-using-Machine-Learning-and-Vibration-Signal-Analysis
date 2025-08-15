import numpy as np
import pandas as pd
import os

# Sampling parameters
fs = 1000  # Hz
t = np.linspace(0, 1, fs, endpoint=False)

# Function to generate different motor fault signals
def generate_signal(freq=50, fault_type="normal", noise_level=0.0):
    # Base sinusoidal signal
    signal = np.sin(2 * np.pi * freq * t)

    # Simulate different faults
    if fault_type == "unbalance":
        signal += 0.5 * np.sin(2 * np.pi * (freq * 2) * t)
    elif fault_type == "bearing":
        signal += 0.3 * np.sin(2 * np.pi * (freq + 120) * t)
    elif fault_type == "broken_rotor":
        signal += 0.4 * np.sin(2 * np.pi * (freq - 20) * t)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, len(t))
    signal += noise

    return signal

# Parameters for dataset generation
frequencies = [50, 60]
fault_types = ["normal", "unbalance", "bearing", "broken_rotor"]
noise_levels = [0.01, 0.05, 0.1]

# Generate dataset
data = []
labels = []

for freq in frequencies:
    for fault in fault_types:
        for noise in noise_levels:
            for _ in range(20):  # number of samples per condition
                sig = generate_signal(freq=freq, fault_type=fault, noise_level=noise)
                data.append(sig)
                labels.append(f"{fault}_{freq}Hz_noise{noise}")

# Convert to DataFrame
df = pd.DataFrame(data)
df["label"] = labels

# Save dataset
os.makedirs("datasets", exist_ok=True)
csv_path = os.path.join("datasets", "motor_signals_diverse.csv")
df.to_csv(csv_path, index=False)

print(f"Dataset saved at: {csv_path}")
print(f"Shape: {df.shape}")