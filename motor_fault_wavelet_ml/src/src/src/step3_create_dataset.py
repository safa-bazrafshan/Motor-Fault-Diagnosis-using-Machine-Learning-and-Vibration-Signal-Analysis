import numpy as np
import pywt
import pandas as pd

def generate_signal(fault=False, fs=1000):
    t = np.linspace(0, 1, fs)
    signal = np.sin(2 * np.pi * 50 * t)
    noise = np.random.normal(0, 0.3, fs)
    signal += noise
    if fault:
        signal[400:600] += 2
    return signal

def extract_wavelet_features(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for coeff in coeffs:
        energy = np.sum(np.square(coeff))
        entropy = -np.sum(np.where(coeff != 0, coeff**2 * np.log(np.abs(coeff) + 1e-12), 0))
        features.extend([energy, entropy])
    return features

num_samples = 20
data = []

for _ in range(num_samples):
    normal_signal = generate_signal(fault=False)
    normal_features = extract_wavelet_features(normal_signal)
    data.append(normal_features + [0])

    faulty_signal = generate_signal(fault=True)
    faulty_features = extract_wavelet_features(faulty_signal)
    data.append(faulty_features + [1])

columns = []
for i in range(5):  # 5 levels of wavelet
    columns.append(f'energy_L{i}')
    columns.append(f'entropy_L{i}')
columns.append('label')

df = pd.DataFrame(data, columns=columns)

df.to_csv("motor_wavelet_dataset.csv", index=False)
print("Dataset saved as motor_wavelet_dataset.csv")
print(df.head())