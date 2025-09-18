import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim

L = 1.0
g = 9.81
m = 1
omega0 = math.sqrt(g / L)
phi0 = math.radians(5.0)
psi0 = 0.0

time = 10.0
nsteps = 1000
dt = time / nsteps
t = np.linspace(0, time, nsteps + 1)

phi_an = phi0 * np.cos(omega0 * t) + psi0 / omega0 * np.sin(omega0 * t)
psi_an = -phi0 * omega0 * np.sin(omega0 * t) + psi0 * np.cos(omega0 * t)

phi_euler, psi_euler = phi0, psi0
phi_rk4, psi_rk4 = phi0, psi0

phi_euler_arr = np.zeros(nsteps + 1)
psi_euler_arr = np.zeros(nsteps + 1)
phi_rk4_arr = np.zeros(nsteps + 1)
psi_rk4_arr = np.zeros(nsteps + 1)

phi_euler_arr[0], psi_euler_arr[0] = phi0, psi0
phi_rk4_arr[0], psi_rk4_arr[0] = phi0, psi0

def get_xy(phi):
    x = L * np.sin(phi)
    y = -L * np.cos(phi)
    return x, y

x_an, y_an = get_xy(phi_an)
x_euler, y_euler = np.zeros(nsteps + 1), np.zeros(nsteps + 1)
x_rk4, y_rk4 = np.zeros(nsteps + 1), np.zeros(nsteps + 1)

x_euler[0], y_euler[0] = get_xy(phi0)
x_rk4[0], y_rk4[0] = get_xy(phi0)

def calculate_energy(phi, psi):
    kinetic = 0.5 * (L * psi) ** 2
    potential = 0.5 * g * L * phi ** 2
    return kinetic + potential

energy_an = calculate_energy(phi_an, psi_an)
energy_euler = np.zeros(nsteps + 1)
energy_rk4 = np.zeros(nsteps + 1)

energy_euler[0] = calculate_energy(phi0, psi0)
energy_rk4[0] = calculate_energy(phi0, psi0)

def euler_step(phi, psi, dt):
    phi_new = phi + psi * dt
    psi_new = psi - omega0**2 * phi * dt
    return phi_new, psi_new

def rk4_step(phi, psi, dt):
    k1_phi = psi
    k1_psi = -omega0**2 * phi
    
    k2_phi = psi + k1_psi * dt / 2
    k2_psi = -omega0**2 * (phi + k1_phi * dt / 2)
    
    k3_phi = psi + k2_psi * dt / 2
    k3_psi = -omega0**2 * (phi + k2_phi * dt / 2)
    
    k4_phi = psi + k3_psi * dt
    k4_psi = -omega0**2 * (phi + k3_phi * dt)
    
    phi_new = phi + (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi) * dt / 6
    psi_new = psi + (k1_psi + 2*k2_psi + 2*k3_psi + k4_psi) * dt / 6
    
    return phi_new, psi_new

for i in range(nsteps):
    phi_euler, psi_euler = euler_step(phi_euler, psi_euler, dt)
    phi_euler_arr[i + 1] = phi_euler
    psi_euler_arr[i + 1] = psi_euler
    energy_euler[i + 1] = calculate_energy(phi_euler, psi_euler)

phi_euler_max = np.max(np.abs(phi_euler_arr)) * 1.2
psi_euler_max = np.max(np.abs(psi_euler_arr)) * 1.2

initial_energy = calculate_energy(phi0, psi0)

phi_euler, psi_euler = phi0, psi0
phi_rk4, psi_rk4 = phi0, psi0
phi_euler_arr = np.zeros(nsteps + 1)
psi_euler_arr = np.zeros(nsteps + 1)
phi_rk4_arr = np.zeros(nsteps + 1)
psi_rk4_arr = np.zeros(nsteps + 1)
phi_euler_arr[0], psi_euler_arr[0] = phi0, psi0
phi_rk4_arr[0], psi_rk4_arr[0] = phi0, psi0
energy_euler = np.zeros(nsteps + 1)
energy_rk4 = np.zeros(nsteps + 1)
energy_euler[0] = initial_energy
energy_rk4[0] = initial_energy

fig = plt.figure(figsize=(16, 12))
plt.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.95, wspace=0.25, hspace=0.35)

ax1 = plt.subplot(2, 2, 1)
ax1.set_xlim(0, time)
ax1.set_ylim(-phi_euler_max, phi_euler_max)
ax1.set_xlabel('Time')
ax1.set_ylabel('phi')
ax1.set_title('Angular Displacement')
ax1.grid(True, alpha=0.3)

traject_an, = ax1.plot(t, phi_an, 'k-', lw=2, label='Analytical')
traject_euler, = ax1.plot([], [], 'r-', lw=1.5, label='Euler')
traject_rk4, = ax1.plot([], [], 'b-', lw=1.5, label='RK4')
ax1.legend()

ax2 = plt.subplot(2, 2, 2)
ax2.set_xlim(-phi_euler_max, phi_euler_max)
ax2.set_ylim(-psi_euler_max, psi_euler_max)
ax2.set_xlabel('phi')
ax2.set_ylabel('psi')
ax2.set_title('Phase trajectory')
ax2.grid(True, alpha=0.3)

phase_an, = ax2.plot(phi_an, psi_an, 'k-', lw=2, label='Analytical')
phase_euler, = ax2.plot([], [], 'r-', lw=1, label='Euler')
phase_rk4, = ax2.plot([], [], 'b-', lw=1, label='RK4')
current_phase, = ax2.plot([], [], 'go', markersize=4)
ax2.legend()

ax3 = plt.subplot(2, 2, 3)
ax3.set_xlim(-L * 1.1, L * 1.1)
ax3.set_ylim(-L * 1.1, L * 0.1)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Trajectory')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

ax3.plot(0, 0, 'ko', markersize=6)

traject_an_xy, = ax3.plot([], [], 'k-', lw=2, label='Analytical')
traject_euler_xy, = ax3.plot([], [], 'r-', lw=1.5, label='Euler')
traject_rk4_xy, = ax3.plot([], [], 'b-', lw=1.5, label='RK4')

current_an, = ax3.plot([], [], 'ko', markersize=4)
current_euler, = ax3.plot([], [], 'ro', markersize=4)
current_rk4, = ax3.plot([], [], 'bo', markersize=4)

pendulum_an, = ax3.plot([], [], 'k--', lw=1, alpha=0.7)
pendulum_euler, = ax3.plot([], [], 'r--', lw=1, alpha=0.7)
pendulum_rk4, = ax3.plot([], [], 'b--', lw=1, alpha=0.7)
ax3.legend()

ax4 = plt.subplot(2, 2, 4)
ax4.set_xlim(0, time)
ax4.set_ylim(initial_energy, initial_energy * 4)
ax4.set_xlabel('Time')
ax4.set_ylabel('E')
ax4.set_title('Energy')
ax4.grid(True, alpha=0.3)

energy_an_line, = ax4.plot([], [], 'k-', lw=2, label='Analytical')
energy_euler_line, = ax4.plot([], [], 'r-', lw=1.5, label='Euler')
energy_rk4_line, = ax4.plot([], [], 'b-', lw=1.5, label='RK4')
ax4.legend()

def init_anim():
    for line in [traject_euler, traject_rk4, phase_euler, phase_rk4, 
                traject_an_xy, traject_euler_xy, traject_rk4_xy,
                energy_an_line, energy_euler_line, energy_rk4_line]:
        line.set_data([], [])
    
    for point in [current_phase, current_an, current_euler, current_rk4]:
        point.set_data([], [])
    
    for pendulum in [pendulum_an, pendulum_euler, pendulum_rk4]:
        pendulum.set_data([], [])
    
    return (traject_euler, traject_rk4, phase_euler, phase_rk4, current_phase,
            traject_an_xy, traject_euler_xy, traject_rk4_xy,
            current_an, current_euler, current_rk4,
            pendulum_an, pendulum_euler, pendulum_rk4,
            energy_an_line, energy_euler_line, energy_rk4_line)

def loop_anim(i):
    global phi_euler, psi_euler, phi_rk4, psi_rk4
    
    phi_euler, psi_euler = euler_step(phi_euler, psi_euler, dt)
    phi_euler_arr[i + 1] = phi_euler
    psi_euler_arr[i + 1] = psi_euler
    x_euler[i + 1], y_euler[i + 1] = get_xy(phi_euler)
    energy_euler[i + 1] = calculate_energy(phi_euler, psi_euler)
    
    phi_rk4, psi_rk4 = rk4_step(phi_rk4, psi_rk4, dt)
    phi_rk4_arr[i + 1] = phi_rk4
    psi_rk4_arr[i + 1] = psi_rk4
    x_rk4[i + 1], y_rk4[i + 1] = get_xy(phi_rk4)
    energy_rk4[i + 1] = calculate_energy(phi_rk4, psi_rk4)
    
    traject_euler.set_data(t[:i + 2], phi_euler_arr[:i + 2])
    traject_rk4.set_data(t[:i + 2], phi_rk4_arr[:i + 2])
    
    phase_euler.set_data(phi_euler_arr[:i + 2], psi_euler_arr[:i + 2])
    phase_rk4.set_data(phi_rk4_arr[:i + 2], psi_rk4_arr[:i + 2])
    current_phase.set_data([phi_rk4_arr[i + 1]], [psi_rk4_arr[i + 1]])
    
    traject_an_xy.set_data(x_an[:i + 2], y_an[:i + 2])
    traject_euler_xy.set_data(x_euler[:i + 2], y_euler[:i + 2])
    traject_rk4_xy.set_data(x_rk4[:i + 2], y_rk4[:i + 2])
    
    current_an.set_data([x_an[i + 1]], [y_an[i + 1]])
    current_euler.set_data([x_euler[i + 1]], [y_euler[i + 1]])
    current_rk4.set_data([x_rk4[i + 1]], [y_rk4[i + 1]])
    
    pendulum_an.set_data([0, x_an[i + 1]], [0, y_an[i + 1]])
    pendulum_euler.set_data([0, x_euler[i + 1]], [0, y_euler[i + 1]])
    pendulum_rk4.set_data([0, x_rk4[i + 1]], [0, y_rk4[i + 1]])
    
    energy_an_line.set_data(t[:i + 2], energy_an[:i + 2])
    energy_euler_line.set_data(t[:i + 2], energy_euler[:i + 2])
    energy_rk4_line.set_data(t[:i + 2], energy_rk4[:i + 2])
    
    return (traject_euler, traject_rk4, phase_euler, phase_rk4, current_phase,
            traject_an_xy, traject_euler_xy, traject_rk4_xy,
            current_an, current_euler, current_rk4,
            pendulum_an, pendulum_euler, pendulum_rk4,
            energy_an_line, energy_euler_line, energy_rk4_line)

ani = anim.FuncAnimation(fig=fig, func=loop_anim, init_func=init_anim, 
                         frames=nsteps, interval=30, repeat=False, blit=True)
plt.show()