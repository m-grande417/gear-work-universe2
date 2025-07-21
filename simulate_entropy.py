
"""
simulate_entropy.py
Scaffold for simulating entropy gradient fields and gear phase misalignment.
"""

import numpy as np
import matplotlib.pyplot as plt

# Simulation grid
GRID_SIZE = (200, 200)
x = np.linspace(-1, 1, GRID_SIZE[0])
y = np.linspace(-1, 1, GRID_SIZE[1])
X, Y = np.meshgrid(x, y)

# Example entropy fields (Gaussian gradients)
S1 = np.exp(-(X**2 + Y**2))
S2 = np.exp(-((X - 0.5)**2 + (Y + 0.5)**2))

# Compute gradient fields
grad_S1 = np.gradient(S1)
grad_S2 = np.gradient(S2)

# Compute angle between gradients (phase misalignment)
dot_product = grad_S1[0]*grad_S2[0] + grad_S1[1]*grad_S2[1]
norm1 = np.sqrt(grad_S1[0]**2 + grad_S1[1]**2)
norm2 = np.sqrt(grad_S2[0]**2 + grad_S2[1]**2)
cos_theta = dot_product / (norm1 * norm2 + 1e-9)
theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

# Plot misalignment
plt.imshow(theta, cmap='inferno', extent=(-1,1,-1,1))
plt.colorbar(label='Phase Misalignment (radians)')
plt.title('Entropy Gear Phase Misalignment')
plt.savefig('data/entropy_misalignment.png')
plt.show()
