import numpy as np
import matplotlib.pyplot as plt

# Define the function fcn
def fcn(k):
    x = (2 * np.pi * np.cos(2 * np.pi * k / 10)) / 10
    if x < -0.4:
        return -0.4
    elif x > 0.4:
        return 0.4
    else:
        return x

# Parameters
L = np.array([[1, 0, 0, -1], [-1, 2, -1, 0], [0, -1, 1, 0], [-1, 0, -1, 2]])
D = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])

# Step Factor Initializations
rho = 0.3
eta = 1
lamda = 0.5
mu = 0.5
epsilon = 10**(-4)
m = 500  # Number of iterations 
n = 10

# Initialize yd
yd1 = np.zeros(n + 1)

for k in range(n + 1):
    yd1[k] = fcn(k)

# Plot results
plt.plot(yd1, 'k-', linewidth=1.5, label='$y_d(k)$')
plt.xlabel('k')
plt.ylabel('$y_d$')
plt.title('yd over iterations')
plt.legend()
plt.grid(True)
plt.show()
