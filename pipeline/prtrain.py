import numpy as np
import matplotlib.pyplot as plt

X = np.load("data/al_rimalll/X_after.npy")
Y = np.load("data/al_rimalll/Y_before.npy")

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

i = 0

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Input AFTER")
plt.imshow(X[i] if X[i].ndim == 2 else X[i, 0], cmap='gray')

plt.subplot(122)
plt.title("Target BEFORE")
plt.imshow(Y[i] if Y[i].ndim == 2 else Y[i, 0], cmap='gray')
plt.show()
