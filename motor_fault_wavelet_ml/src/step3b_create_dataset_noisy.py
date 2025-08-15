import pandas as pd

# Normal (noisy) features
normal_energy = [65.57, 8.31, 418.97, 122.04, 55.54, 119.93]
normal_entropy = [-118.15, 6.20, -915.22, -47.66, 47.97, 82.92]

# Faulty (noisy) features
faulty_energy = [196.47, 11.50, 480.84, 433.22, 283.27, 124.34]
faulty_entropy = [-639.02, 4.42, -1109.98, -755.22, -166.08, 83.20]

# Combine features
features = [
    normal_energy + normal_entropy,  # label 0
    faulty_energy + faulty_entropy   # label 1
]

labels = [0, 1]  # 0: normal, 1: faulty

# Create DataFrame
columns = [f'energy_L{i}' for i in range(6)] + [f'entropy_L{i}' for i in range(6)]
df = pd.DataFrame(features, columns=columns)
df['label'] = labels

# Save to CSV
df.to_csv("motor_wavelet_dataset_noisy.csv", index=False)

print("Dataset saved as motor_wavelet_dataset_noisy.csv")
print(df)