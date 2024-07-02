import numpy as np
import matplotlib.pyplot as plt


# Define the function fcn
def fcn_yd1(k):
    x = (2 * np.pi  * np.cos(2 * np.pi * k / 10)) 
    if x < -0.4:
        return -0.4
    elif x > 0.4:
        return 0.4
    else:
        return x
    
def fcn_yd2(k):
    x = (2 * np.pi  * np.cos(2 * np.pi * k / 7)) 
    if x < -0.6:
        return -0.6
    elif x > 0.6:
        return 0.6
    else:
        return x
    
def fcn_yd3(k):
    x = (2 * np.pi  * np.cos(2 * np.pi * k / 5)) 
    if x < -0.8:
        return -0.8
    elif x > 0.8:
        return 0.8
    else:
        return x

def fcn_phi1(k):
    for k in range(n):
       
        if k == 1:
            x = 2
            return x[1]   
        elif k == 2:
            x = phi1[k-1] + eta * ( y1[k] - 0 -(phi1[k-1] * y1[k-1] - 0) - phi2[k-1] * np.dot((umk1[k-1] - 0).T ) ) * np.dot((y1[k-1] - 0).T) / mu + abs(y1[k-1]-0)**2 + abs(umk1[k-1]-0)**2
            return x
        else:
            x =  phi1[k-1] + eta * ( y1[k] - y1[k-1] -(phi1[k-1] * y1[k-1] - y1[k-2]) - phi2[k-1] * np.dot((umk1[k-1] - umk1[k-2]).T ) ) * np.dot((y1[k-1] - y1[k-2]).T) / mu + abs(y1[k-1]-y1[k-2])**2 + abs(umk1[k-1]-umk1[k-2])**2
        

def fcn_phi2(k):
    for k in range(n):
       
        if k == 1:
            x = 2
            return x[1]   
        elif k == 2:
            x = phi2[k-1] + eta * ( y2[k] - 0 -(phi2[k-1] * y2[k-1] - 0) - phi2[k-1] * np.dot((umk2[k-1] - 0).T ) ) * np.dot((y2[k-1] - 0).T) / mu + abs(y2[k-1]-0)**2 + abs(umk2[k-1]-0)**2
            return x
        else:
            x =  phi2[k-1] + eta * ( y2[k] - y2[k-1] -(phi1[k-1] * y2[k-1] - y2[k-2]) - phi2[k-1] * np.dot((umk2[k-1] - umk2[k-2]).T ) ) * np.dot((y2[k-1] - y2[k-2]).T) / mu + abs(y2[k-1]-y2[k-2])**2 + abs(umk2[k-1]-umk2[k-2])**2


def fcn_umk1(k):
    umk1 = umk1[k-1] + ((rho * np.dot(phi2[k].T *(yd1[k+1]) - y1[k] - phi1[k] *  y1[k] - y1[k-1])))
    return umk1

def fcn_umk1(k):
    umk2 = umk2[k-1] + ((rho * np.dot(phi2[k].T *(yd2[k+1]) - y2[k] - phi1[k] *  y2[k] - y2[k-1])))
    return umk2

def fcn_umk3(k):
    umk3 = umk3[k-1] + ((rho * np.dot(phi2[k].T *(yd3[k+1]) - y3[k] - phi3[k] *  y3[k] - y3[k-1])))
    return umk3

def fcn_usk1(k):
    usk = phi2[k]**-1 * (yd1[k+1]) - phi1[k] * y1[k] - y1[k-1] - y1[k] - (1 - 1*T) * yd1[k] - y1[k] + epsilon * T * np.sign(yd1[k] - y1[k])
    return usk

def fcn_usk2(k):
    usk = phi2[k]**-1 * (yd2[k+1]) - phi1[k] * y2[k] - y2[k-1] - y2[k] - (1 - 1*T) * yd2[k] - y2[k] + epsilon * T * np.sign(yd2[k] - y2[k])
    return usk

def fcn_usk3(k):
    usk = phi2[k]**-1 * (yd3[k+1]) - phi1[k] * y3[k] - y3[k-1] - y3[k] - (1 - 1*T) * yd3[k] - y3[k] + epsilon * T * np.sign(yd3[k] - y3[k])
    return usk


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

# Initialize yd
yd1 = np.zeros(n + 1)
yd2 = np.zeros(n + 1)
yd3 = np.zeros(n + 1)

#Initialize umk
umk1 = np.zeros(n + 1)
umk2 = np.zeros(n + 1)
umk3 = np.zeros(n + 1)

# Define phi as arrays
phi1 = np.zeros((n, 1))
phi2 = np.zeros((n, 1))
phi3 = np.zeros((n, 1))


for k in range(n + 1):
    yd1[k] = fcn_yd1(k)
    yd2[k] = fcn_yd2(k)
    yd3[k] = fcn_yd3(k)

    umk[k] = fcn_umk(k)

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
