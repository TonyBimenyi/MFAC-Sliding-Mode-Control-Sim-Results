import numpy as np
import matplotlib.pyplot as plt

# Define the function fcn
def fcn_yd1(k):
    x = (2 * np.pi /10 * np.cos(2 * np.pi * k / 10)) 
    if x < -0.4:
        return -0.4
    elif x > 0.4:
        return 0.4
    else:
        return x
    
def fcn_yd2(k):
    x = (2 * np.pi /7 * np.cos(2 * np.pi * k / 7)) 
    if x < -0.6:
        return -0.6
    elif x > 0.6:
        return 0.6
    else:
        return x
    
def fcn_yd3(k):
    x = (2 * np.pi /5 * np.cos(2 * np.pi * k / 5)) 
    if x < -0.8:
        return -0.8
    elif x > 0.8:
        return 0.8
    else:
        return x

# Parameters
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
yd2 = np.zeros(n + 1)
yd3 = np.zeros(n + 1)

for k in range(n + 1):
    yd1[k] = fcn_yd1(k)
    yd2[k] = fcn_yd2(k)
    yd3[k] = fcn_yd3

# Plot results using subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot yd1
axs[0].plot(yd1, 'k-', linewidth=1.5, label='$y_d1(k)$')
axs[0].set_xlabel('k')
axs[0].set_ylabel('$y_d1$')
axs[0].set_title('yd1 over iterations')
axs[0].legend()
axs[0].grid(True)

# Plot yd2
axs[1].plot(yd2, 'r-', linewidth=1.5, label='$y_d2(k)$')
axs[1].set_xlabel('k')
axs[1].set_ylabel('$y_d2$')
axs[1].set_title('yd2 over iterations')
axs[1].legend()
axs[1].grid(True)

# Plot yd3
axs[2].plot(yd3, 'b-', linewidth=1.5, label='$y_d3(k)$')
axs[2].set_xlabel('k')
axs[2].set_ylabel('$y_d3$')
axs[2].set_title('yd3 over iterations')
axs[2].legend()
axs[2].grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
