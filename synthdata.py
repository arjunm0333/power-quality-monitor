import numpy as np

samples_per_class = 600
timesteps = 1000
fs = 1000
t = np.linspace(0, 1, timesteps)

def generate_signal(amplitude=1, freq=50, harmonic=0):
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    if harmonic > 0:
        signal += harmonic * np.sin(2 * np.pi * 3 * freq * t)
    return signal

data = []
labels = []

# Normal
for _ in range(samples_per_class):
    amp = np.random.uniform(0.95, 1.05)
    data.append(generate_signal(amp))
    labels.append(0)

# Sag
for _ in range(samples_per_class):
    amp = np.random.uniform(0.5, 0.8)
    data.append(generate_signal(amp))
    labels.append(1)

# Swell
for _ in range(samples_per_class):
    amp = np.random.uniform(1.2, 1.5)
    data.append(generate_signal(amp))
    labels.append(2)

# Harmonic
for _ in range(samples_per_class):
    harm = np.random.uniform(0.2, 0.5)
    data.append(generate_signal(1, harmonic=harm))
    labels.append(3)

X = np.array(data)
y = np.array(labels)

np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset created successfully.")