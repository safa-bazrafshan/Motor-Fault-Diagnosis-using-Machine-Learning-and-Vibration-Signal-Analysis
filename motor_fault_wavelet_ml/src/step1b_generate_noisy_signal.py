import numpy as np
import matplotlib.pyplot as plt

def generate_signal(freq=50, t_end=1.0, sample_rate=1000):
    t = np.linspace(0, t_end, int(t_end * sample_rate))
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal

def add_noise(signal, noise_level=0.5):
    noise = noise_level * np.random.randn(len(signal))
    return signal + noise

if __name__ == "__main__":
    t, normal = generate_signal(freq=50)
    _, faulty = generate_signal(freq=50)
    _, harmonic = generate_signal(freq=120)

    faulty += harmonic  # fault signal = 50 Hz + 120 Hz

    normal_noisy = add_noise(normal)
    faulty_noisy = add_noise(faulty)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, normal_noisy)
    plt.title("Noisy Normal Signal")

    plt.subplot(2, 1, 2)
    plt.plot(t, faulty_noisy)
    plt.title("Noisy Faulty Signal")

    plt.tight_layout()
    plt.show()  # just show the figure, you will save it manually