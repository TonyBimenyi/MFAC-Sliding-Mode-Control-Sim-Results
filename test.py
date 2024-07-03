import numpy as np
import matplotlib.pyplot as plt

# Define the function fcn
def fcn_yd1(k):
    x = 2 * np.pi * np.cos(2 * np.pi * k / 10)
    return np.clip(x, -0.4, 0.4)

def fcn_yd2(k):
    x = 2 * np.pi * np.cos(2 * np.pi * k / 7)
    return np.clip(x, -0.6, 0.6)

def fcn_yd3(k):
    x = 2 * np.pi * np.cos(2 * np.pi * k / 5)
    return np.clip(x, -0.8, 0.8)

def fcn_phi1(k):
    if k == 1:
        return 0
    elif k == 2:
        return phi1[k-1] + eta * (y1[k] - 0 - (phi1[k-1] * y1[k-1] - 0) - phi2[k-1] * (umk1[k-1] - 0)) * (y1[k-1] - 0) / (mu + abs(y1[k-1]-0)**2 + abs(umk1[k-1]-0)**2)
    else:
        return phi1[k-1] + eta * (y1[k] - y1[k-1] - (phi1[k-1] * y1[k-1] - y1[k-2]) - phi2[k-1] * (umk1[k-1] - umk1[k-2])) * (y1[k-1] - y1[k-2]) / (mu + abs(y1[k-1]-y1[k-2])**2 + abs(umk1[k-1]-umk1[k-2])**2)

def fcn_phi2(k):
    if k == 1:
        return 0
    elif k == 2:
        return phi2[k-1] + eta * (y2[k] - 0 - (phi2[k-1] * y2[k-1] - 0) - phi2[k-1] * (umk2[k-1] - 0)) * (y2[k-1] - 0) / (mu + abs(y2[k-1]-0)**2 + abs(umk2[k-1]-0)**2)
    else:
        return phi2[k-1] + eta * (y2[k] - y2[k-1] - (phi1[k-1] * y2[k-1] - y2[k-2]) - phi2[k-1] * (umk2[k-1] - umk2[k-2])) * (y2[k-1] - y2[k-2]) / (mu + abs(y2[k-1]-y2[k-2])**2 + abs(umk2[k-1]-umk2[k-2])**2)

def fcn_phi3(k):
    if k == 1:
        return 0
    elif k == 2:
        return phi3[k-1] + eta * (y3[k] - 0 - (phi3[k-1] * y3[k-1] - 0) - phi3[k-1] * (umk3[k-1] - 0)) * (y3[k-1] - 0) / (mu + abs(y3[k-1]-0)**2 + abs(umk3[k-1]-0)**2)
    else:
        return phi3[k-1] + eta * (y3[k] - y3[k-1] - (phi3[k-1] * y3[k-1] - y3[k-2]) - phi3[k-1] * (umk3[k-1] - umk3[k-2])) * (y3[k-1] - y3[k-2]) / (mu + abs(y3[k-1]-y3[k-2])**2 + abs(umk3[k-1]-umk3[k-2])**2)

def fcn_umk1(k):
    if k == 1:
       return 0
    else:
         return umk1[k-1] + (rho * (phi2[k].T * (yd1[k+1]) - y1[k] - phi1[k] * y1[k] - y1[k-1]))

def fcn_umk2(k):
    
    if k == 1:
       return 0
        
    else:
        return umk2[k-1] + (rho * (phi2[k].T * (yd2[k+1]) - y2[k] - phi1[k] * y2[k] - y2[k-1]))

def fcn_umk3(k):
    if k == 1:
        return 0
    else:
        return umk3[k-1] + (rho * (phi2[k].T * (yd3[k+1]) - y3[k] - phi3[k] * y3[k] - y3[k-1]))

def fcn_usk1(k):
    if k == 0:
        return 0
    else:
        return (1 / phi2[k][0] if phi2[k][0] != 0 else 0) * (yd1[k+1]) - phi1[k][0] * y1[k][0] - y1[k-1][0] - y1[k][0] - (1 - 1*T) * yd1[k] - y1[k][0] + epsilon * T * np.sign(yd1[k] - y1[k][0])

def fcn_usk2(k):
    if k < n:
        return (1 / phi2[k][0] if phi2[k][0] != 0 else 0) * (yd2[k+1]) - phi1[k][0] * y2[k][0] - y2[k-1][0] - y2[k][0] - (1 - 1*T) * yd2[k] - y2[k][0] + epsilon * T * np.sign(yd2[k] - y2[k][0])
    else:
        return usk2[k-1]

def fcn_usk3(k):
    if k < n:
        return (1 / phi2[k][0] if phi2[k][0] != 0 else 0) * (yd3[k+1]) - phi1[k][0] * y3[k][0] - y3[k-1][0] - y3[k][0] - (1 - 1*T) * yd3[k] - y3[k][0] + epsilon * T * np.sign(yd3[k] - y3[k][0])
    else:
        return usk3[k-1]

def fcn_u1():
    return umk1 + gamma * usk1

def fcn_u2():
    return umk2 + gamma * usk2

def fcn_u3():
    return umk3 + gamma * usk3

def fcn_y1(k):
    return u1[k][0] + np.sin(y1[k][0])

def fcn_y2(k):
    return u2[k][0] + np.sin(y2[k][0])

def fcn_y3(k):
    return u3[k][0] + np.sin(y3[k][0])

# Parameters
# Step Factor Initializations
rho = 0.3
eta = 1
lamda = 0.5
mu = 0.5
epsilon = 10
m = 500  # Number of iterations 
n = 10
T = 0.0001
q = 100
d = [0.1, 0.1, 0.1, 0.2, 0.2]
gamma = np.diag(d)

# Initialize yd
yd1 = np.zeros(n + 1)
yd2 = np.zeros(n + 1)
yd3 = np.zeros(n + 1)

# Initialize umk
umk1 = np.zeros(n + 1)
umk2 = np.zeros(n + 1)
umk3 = np.zeros(n + 1)

# Initialize u
u1 = np.zeros((n, 1))
u2 = np.zeros((n, 1))
u3 = np.zeros((n, 1))

usk1 = np.zeros(n + 1)
usk2 = np.zeros(n + 1)
usk3 = np.zeros(n + 1)

# Define phi as arrays
phi1 = np.zeros((n, 1))
phi2 = np.zeros((n, 1))
phi3 = np.zeros((n, 1))

# Define y as arrays
y1 = np.zeros((n + 1, 1))
y2 = np.zeros((n + 1, 1))
y3 = np.zeros((n + 1, 1))

for k in range(n):
    yd1[k] = fcn_yd1(k)
    yd2[k] = fcn_yd2(k)
    yd3[k] = fcn_yd3(k)
    
    phi1[k] = fcn_phi1(k)
    phi2[k] = fcn_phi2(k)
    phi3[k] = fcn_phi3(k)

    umk1[k] = fcn_umk1(k)
    umk2[k] = fcn_umk2(k)
    umk3[k] = fcn_umk3(k)
    
    usk1[k] = fcn_usk1(k)
    usk2[k] = fcn_usk2(k)
    usk3[k] = fcn_usk3(k)
    
    y1[1] = 0.51
    y2[1] = 2.5
    y3[1] = 3.5
    
    y1[k+1] = fcn_y1(k)
    y2[k+1] = fcn_y2(k)
    y3[k+1] = fcn_y3(k)

# Plot results using subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot yd1
axs[0].plot(yd1, 'k-', linewidth=1.5, label='$y_d1(k)$')
axs[0].plot(y1, 'k-', linewidth=1.5, label='$y_d1(k)$')
axs[0].set_xlabel('k')
axs[0].set_ylabel('$y_d1$')
axs[0].set_title('yd1 over iterations')
axs[0].legend()
axs[0].grid(True)

# Plot yd2
axs[1].plot(yd2, 'r-', linewidth=1.5, label='$y_d2(k)$')
axs[1].plot(y2, 'r-', linewidth=1.5, label='$y_d2(k)$')
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
