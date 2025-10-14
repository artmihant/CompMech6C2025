import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Model parameters
a1 = 5.0
b1 = 3.0
a2 = 0.1
b2 = 2.0
d1 = 0.4
d2 = 0.01

# Calculation
x_0 = 0.5
y_0 = 0.1
z_0 = 0.1
sol_0 = np.array((x_0, y_0, z_0), dtype=np.float32)

# Time mesh
t_0 = 0.0
t_1 = 100.0
N = 3000
t = np.linspace(t_0, t_1, N + 1)
tau = (t_1 - t_0) / N

def f1(u):
    return a1 * u / (1.0 + b1 * u)

def f2(u):
    return a2 * u / (1.0 + b1 * u) 

# Differential equation:
# x' = x * (1-x/K) - f1(x)*y 
# y' = f1(x)*y - f2(y)*z - d1*y
# z' = f2(y)*z - d2*z
def f(w):
    x = w[0]
    y = w[1]
    z = w[2]
    return np.array([x * (1.0 - x) - f1(x) * y, 
                    f1(x) * y - f2(y) * z - d1 * y, 
                    f2(y) * z - d2 * z])

def calc_food_chain(sol_0):
    sol = np.zeros((N + 1, 3), dtype=np.float32)
    sol[0] = sol_0
        
    for i in range(0, N):
        d1 = tau * f(sol[i])
        d2 = tau * f(sol[i] + d1/2)
        d3 = tau * f(sol[i] + d2/2)
        d4 = tau * f(sol[i] + d3)
        sol[i+1] = sol[i] + (d1 + 2*d2 + 2*d3 + d4)/6.0         
    
    return sol


sol_rk4 = calc_food_chain(sol_0)

# Create figure with subplots
fig = plt.figure(figsize=(15, 12))

# Time plot
plt.subplot(2, 2, 1)
plt.plot(t, sol_rk4[:, 0], 'b-', label='Prey (X)', linewidth=2)
plt.plot(t, sol_rk4[:, 1], 'r-', label='Predator (Y)', linewidth=2)
plt.plot(t, sol_rk4[:, 2], 'g-', label='Superpredator (Z)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Population Density')
plt.title('Time Series of Three-Species Food Chain')
plt.legend()
plt.grid(True, alpha=0.3)

# Phase portraits
plt.subplot(2, 2, 2)
plt.plot(sol_rk4[:, 0], sol_rk4[:, 1], 'purple', linewidth=1)
plt.scatter(sol_rk4[0, 0], sol_rk4[0, 1], color='red', s=100, label='Start', zorder=5)
plt.scatter(sol_rk4[-1, 0], sol_rk4[-1, 1], color='blue', s=100, label='End', zorder=5)
plt.xlabel('Prey (X)')
plt.ylabel('Predator (Y)')
plt.title('Phase Portrait: X vs Y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(sol_rk4[:, 1], sol_rk4[:, 2], 'orange', linewidth=1)
plt.scatter(sol_rk4[0, 1], sol_rk4[0, 2], color='red', s=100, label='Start', zorder=5)
plt.scatter(sol_rk4[-1, 1], sol_rk4[-1, 2], color='blue', s=100, label='End', zorder=5)
plt.xlabel('Predator (Y)')
plt.ylabel('Superpredator (Z)')
plt.title('Phase Portrait: Y vs Z')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(sol_rk4[:, 0], sol_rk4[:, 2], 'brown', linewidth=1)
plt.scatter(sol_rk4[0, 0], sol_rk4[0, 2], color='red', s=100, label='Start', zorder=5)
plt.scatter(sol_rk4[-1, 0], sol_rk4[-1, 2], color='blue', s=100, label='End', zorder=5)
plt.xlabel('Prey (X)')
plt.ylabel('Superpredator (Z)')
plt.title('Phase Portrait: X vs Z')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analysis of periodic oscillations
print("=== ANALYSIS OF PERIODIC OSCILLATIONS ===")

# Check for periodicity by analyzing peaks in the last half of the simulation
half_idx = N // 2
x_half = sol_rk4[half_idx:, 0]
y_half = sol_rk4[half_idx:, 1]
z_half = sol_rk4[half_idx:, 2]

# Find local maxima
from scipy.signal import find_peaks

for i, (pop, name) in enumerate(zip([x_half, y_half, z_half], ['X', 'Y', 'Z'])):
    peaks, _ = find_peaks(pop, height=np.mean(pop))
    
    if len(peaks) > 1:
        periods = np.diff(t[half_idx:][peaks])
        mean_period = np.mean(periods)
        std_period = np.std(periods)
        print(f"\n{name} - Found {len(peaks)} peaks")
        print(f"  Mean period: {mean_period:.3f} ± {std_period:.3f}")
        
        if std_period < 0.5:  # Threshold for considering as periodic
            print(f"  → Suggests periodic behavior for {name}")
        else:
            print(f"  → Suggests irregular/chaotic behavior for {name}")
    else:
        print(f"\n{name} - Insufficient peaks found for period analysis")
