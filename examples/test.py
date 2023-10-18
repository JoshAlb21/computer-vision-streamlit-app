import numpy as np
import matplotlib.pyplot as plt

# Generate example data
t = np.linspace(0, 2 * np.pi, 100)
x = np.sin(t)
y = np.sin(2 * t)

# Plot parametric curve
plt.plot(x, y)
plt.scatter(x, y, c=t, cmap='viridis')  # Color points by parameter t
plt.title('Parametric Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()