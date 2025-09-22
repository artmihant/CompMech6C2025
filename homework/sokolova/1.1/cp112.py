import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

L = 1.0
g = 9.81
omega0 = math.sqrt(g / L)
phi0 = math.radians(60.0) 
psi0 = 0.0

time = 10.0
nsteps = 1000
dt = time / nsteps
t = np.linspace(0, time, nsteps + 1)

def get_xy(phi, L_val=L):
    x = L_val * np.sin(phi)
    y = -L_val * np.cos(phi)
    return x, y

def rk4_step(phi, psi, dt, L_val=L):
    def derivs(phi, psi):
        dphi = psi
        dpsi = -(g / L_val) * np.sin(phi)
        return dphi, dpsi
    
    k1_phi, k1_psi = derivs(phi, psi)
    k2_phi, k2_psi = derivs(phi + k1_phi * dt/2, psi + k1_psi * dt/2)
    k3_phi, k3_psi = derivs(phi + k2_phi * dt/2, psi + k2_psi * dt/2)
    k4_phi, k4_psi = derivs(phi + k3_phi * dt, psi + k3_psi * dt)
    
    phi_new = phi + (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi) * dt / 6
    psi_new = psi + (k1_psi + 2*k2_psi + 2*k3_psi + k4_psi) * dt / 6
    
    return phi_new, psi_new

fig = plt.figure(figsize=(18, 6))
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.9, 
                    wspace=0.25, hspace=0.3)

ax1 = plt.subplot(1, 3, 1)
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Траектория маятника')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.plot(0, 0, 'ko', markersize=8)
line_traj, = ax1.plot([], [], 'g-', lw=1.5, alpha=0.7)
line_pend, = ax1.plot([], [], 'o-', lw=2, color='blue', markersize=8)

ax2 = plt.subplot(1, 3, 2)
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Система гармонически-сопряженных маятников')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.plot(0, 0, 'ko', markersize=8)

ax3 = plt.subplot(1, 3, 3)
ax3.set_xlim(-np.pi, np.pi)
ax3.set_ylim(-5, 5)
ax3.set_xlabel('Угол (рад)')
ax3.set_ylabel('Угловая скорость (рад/с)')
ax3.set_title('Фазовые портреты для разных начальных условий')
ax3.grid(True, alpha=0.3)

num_pendulums = 3
lengths = [L, L/4, L/9]
colors = ['red', 'green', 'blue']
pendulum_lines = []
pendulum_trails = []

for i in range(num_pendulums):
    line, = ax2.plot([], [], 'o-', lw=2, color=colors[i], markersize=6, 
                    label=f'L={lengths[i]:.2f}')
    trail, = ax2.plot([], [], '-', lw=1, color=colors[i], alpha=0.5)
    pendulum_lines.append(line)
    pendulum_trails.append(trail)

ax2.legend(loc='upper right')

initial_angles = [math.radians(30), math.radians(60), math.radians(90), math.radians(120)]
initial_velocities = [0, 1, 2, 3]

for i, angle in enumerate(initial_angles):
    for j, velocity in enumerate(initial_velocities):
        phi_temp, psi_temp = angle, velocity
        phi_arr, psi_arr = [], []
        
        for _ in range(500):
            phi_temp, psi_temp = rk4_step(phi_temp, psi_temp, 0.05)
            phi_arr.append(phi_temp)
            psi_arr.append(psi_temp)
        
        ax3.plot(phi_arr, psi_arr, '-', lw=0.5, 
                color=plt.cm.viridis((i * len(initial_velocities) + j) / (len(initial_angles) * len(initial_velocities))))

phi = phi0
psi = psi0
phi_history = np.zeros(nsteps + 1)
psi_history = np.zeros(nsteps + 1)
energy_history = np.zeros(nsteps + 1)
x_history = np.zeros(nsteps + 1)
y_history = np.zeros(nsteps + 1)

pendulums_phi = [phi0] * num_pendulums
pendulums_psi = [psi0] * num_pendulums
pendulums_x = [[] for _ in range(num_pendulums)]
pendulums_y = [[] for _ in range(num_pendulums)]

def init():
    line_traj.set_data([], [])
    line_pend.set_data([], [])
    
    for line in pendulum_lines:
        line.set_data([], [])
    for trail in pendulum_trails:
        trail.set_data([], [])
    
    return (line_traj, line_pend, *pendulum_lines, *pendulum_trails)

def update(frame):
    global phi, psi, pendulums_phi, pendulums_psi
    
    phi, psi = rk4_step(phi, psi, dt)
    phi_history[frame] = phi
    psi_history[frame] = psi
    x, y = get_xy(phi)
    x_history[frame] = x
    y_history[frame] = y
    
    for i in range(num_pendulums):
        pendulums_phi[i], pendulums_psi[i] = rk4_step(
            pendulums_phi[i], pendulums_psi[i], dt, lengths[i])
        x_i, y_i = get_xy(pendulums_phi[i], lengths[i])
        pendulums_x[i].append(x_i)
        pendulums_y[i].append(y_i)
        
        if len(pendulums_x[i]) > 100:
            pendulums_x[i] = pendulums_x[i][-100:]
            pendulums_y[i] = pendulums_y[i][-100:]
    
    line_traj.set_data(x_history[:frame], y_history[:frame])
    line_pend.set_data([0, x], [0, y])
    
    for i in range(num_pendulums):
        x_i, y_i = get_xy(pendulums_phi[i], lengths[i])
        pendulum_lines[i].set_data([0, x_i], [0, y_i])
        pendulum_trails[i].set_data(pendulums_x[i], pendulums_y[i])
    
    return (line_traj, line_pend, *pendulum_lines, *pendulum_trails)

ani = animation.FuncAnimation(fig, update, frames=nsteps, init_func=init, 
                              interval=30, blit=True)

plt.show()