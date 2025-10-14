import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Model parameters
sigma = 10.0
rho = 28.0
beta = 8.0/3.0

# Initial conditions
x_0 = 1.0
y_0 = 2.0
z_0 = 3.0
sol_0 = np.array((x_0, y_0, z_0), dtype=np.float32)
x_0_perturbed = 1.01 * x_0
sol_0_perturbed = np.array((x_0_perturbed, y_0, z_0), dtype=np.float32)

# Time mesh
t_0 = 0.0
t_1 = 30.0
N = 3000
t = np.linspace(t_0, t_1, N + 1)
tau = (t_1 - t_0) / N

# Differential equation:
# x' = sigma * (y - x)
# y' = x * (rho - z) - y
# z' = x * y - beta * z
def f(w):
    x = w[0]
    y = w[1]
    z = w[2]
    return np.array((sigma * (y - x), x * (rho - z) - y, x * y - beta * z))

def calc_lorenz(sol_0):
    sol = np.zeros((N + 1, 3), dtype=np.float32)
    sol[0] = sol_0
        
    for i in range(0, N):
        d1 = tau * f(sol[i])
        d2 = tau * f(sol[i] + d1/2)
        d3 = tau * f(sol[i] + d2/2)
        d4 = tau * f(sol[i] + d3)
        sol[i+1] = sol[i] + (d1 + 2*d2 + 2*d3 + d4)/6.0         
    
    return sol

# Calculation
sol_rk4 = calc_lorenz(sol_0)
sol_rk4_perturbed = calc_lorenz(sol_0_perturbed)

# Phase portraits
fig = plt.figure(figsize=(20, 15))

ax1 = plt.subplot(2, 3, 1)
ax1.plot(sol_rk4[:, 0], sol_rk4[:, 1], 'b-', linewidth=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Phase portrait X-Y')
ax1.grid(True)

ax2 = plt.subplot(2, 3, 2)
ax2.plot(sol_rk4[:, 0], sol_rk4[:, 2], 'r-', linewidth=0.5)
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax2.set_title('Phase portrait X-Z')
ax2.grid(True)

ax3 = plt.subplot(2, 3, 3)
ax3.plot(sol_rk4[:, 1], sol_rk4[:, 2], 'g-', linewidth=0.5)
ax3.set_xlabel('y')
ax3.set_ylabel('z')
ax3.set_title('Phase portrait Y-Z')
ax3.grid(True)

# Time plot
ax4 = plt.subplot(2, 3, 4)
ax4.plot(t, sol_rk4[:, 0], 'b-', label='x(t)', linewidth=1)
ax4.plot(t, sol_rk4[:, 1], 'r-', label='y(t)', linewidth=1)
ax4.plot(t, sol_rk4[:, 2], 'g-', label='z(t)', linewidth=1)
ax4.set_xlabel('t')
ax4.set_ylabel('x/y/z')
ax4.set_title('Time plot')
ax4.legend()
ax4.grid(True)

# Compare two trajectories
difference = np.linalg.norm(sol_rk4 - sol_rk4_perturbed, axis=1)

ax5 = plt.subplot(2, 3, 5)
ax5.plot(t, sol_rk4[:, 0], 'b-', linewidth=1, label='Initial x(t)')
ax5.plot(t, sol_rk4_perturbed[:, 0], 'r--', linewidth=1, label='Pertrubed x(t)')
ax5.set_xlabel('t')
ax5.set_ylabel('x')
ax5.set_title('Two trajectories comparison')
ax5.legend()
ax5.grid(True)

ax6 = plt.subplot(2, 3, 6)
ax6.semilogy(t, difference, 'k-', linewidth=1)
ax6.set_xlabel('t')
ax6.set_ylabel('||s_1 - s_2||')
ax6.set_title('Trajectories difference (log scale)')
ax6.grid(True)

plt.tight_layout()
plt.show()
