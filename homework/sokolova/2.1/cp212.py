import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.integrate import solve_ivp

L1 = 1
L2 = 2
g = 9.81
w1 = np.sqrt(g/L1)
w2 = np.sqrt(g/L2)

phi1_0 = np.radians(60.0)
phi2_0 = np.radians(0.0)
psi1_0 = 0.0
psi2_0 = 0.0

time = 30.0
nsteps = 200
dt = time / nsteps
t = np.linspace(0, time, nsteps + 1)

phi1_arr = np.zeros(nsteps + 1)
phi2_arr = np.zeros(nsteps + 1)
psi1_arr = np.zeros(nsteps + 1)
psi2_arr = np.zeros(nsteps + 1)

phi1_arr[0], phi2_arr[0] = phi1_0, phi2_0
psi1_arr[0], psi2_arr[0] = psi1_0, psi2_0

def equations_of_motion(phi1, phi2, psi1, psi2):
    psi1_dot = -2.0 * g * phi1 / L1 + g * phi2 / L1
    psi2_dot = -2.0 * g * phi2 / L2 + 2.0 * g * phi1 / L2
    return psi1, psi2, psi1_dot, psi2_dot

def rk4_step(phi1, phi2, psi1, psi2, dt):
    k1_phi1, k1_phi2, k1_psi1, k1_psi2 = equations_of_motion(phi1, phi2, psi1, psi2)
    
    phi1_temp = phi1 + k1_phi1 * dt / 2
    phi2_temp = phi2 + k1_phi2 * dt / 2
    psi1_temp = psi1 + k1_psi1 * dt / 2
    psi2_temp = psi2 + k1_psi2 * dt / 2
    k2_phi1, k2_phi2, k2_psi1, k2_psi2 = equations_of_motion(phi1_temp, phi2_temp, psi1_temp, psi2_temp)
    
    phi1_temp = phi1 + k2_phi1 * dt / 2
    phi2_temp = phi2 + k2_phi2 * dt / 2
    psi1_temp = psi1 + k2_psi1 * dt / 2
    psi2_temp = psi2 + k2_psi2 * dt / 2
    k3_phi1, k3_phi2, k3_psi1, k3_psi2 = equations_of_motion(phi1_temp, phi2_temp, psi1_temp, psi2_temp)
    
    phi1_temp = phi1 + k3_phi1 * dt
    phi2_temp = phi2 + k3_phi2 * dt
    psi1_temp = psi1 + k3_psi1 * dt
    psi2_temp = psi2 + k3_psi2 * dt
    k4_phi1, k4_phi2, k4_psi1, k4_psi2 = equations_of_motion(phi1_temp, phi2_temp, psi1_temp, psi2_temp)
    
    phi1_new = phi1 + (k1_phi1 + 2*k2_phi1 + 2*k3_phi1 + k4_phi1) * dt / 6
    phi2_new = phi2 + (k1_phi2 + 2*k2_phi2 + 2*k3_phi2 + k4_phi2) * dt / 6
    psi1_new = psi1 + (k1_psi1 + 2*k2_psi1 + 2*k3_psi1 + k4_psi1) * dt / 6
    psi2_new = psi2 + (k1_psi2 + 2*k2_psi2 + 2*k3_psi2 + k4_psi2) * dt / 6
    
    return phi1_new, phi2_new, psi1_new, psi2_new

phi1, phi2, psi1, psi2 = phi1_0, phi2_0, psi1_0, psi2_0

for i in range(nsteps):
    phi1, phi2, psi1, psi2 = rk4_step(phi1, phi2, psi1, psi2, dt)
    phi1_arr[i + 1] = phi1
    phi2_arr[i + 1] = phi2
    psi1_arr[i + 1] = psi1
    psi2_arr[i + 1] = psi2

def f(t, x):
    phi1, phi2, psi1, psi2 = x
    return np.array([psi1, psi2, -2.0*g*phi1/L1 + g*phi2/L1, -2.0*g*phi2/L2 + 2.0*g*phi1/L2])

sol_rk45_result = solve_ivp(f, [0, time], [phi1_0, phi2_0, psi1_0, psi2_0], 
                           method='RK45', t_eval=t, rtol=1e-6, atol=1e-8)
sol_rk45 = sol_rk45_result.y.T

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(t, np.degrees(phi1_arr), 'r-', label='φ1 (первый маятник)')
ax1.plot(t, np.degrees(phi2_arr), 'b-', label='φ2 (второй маятник)')
ax1.set_xlabel('Время (с)')
ax1.set_ylabel('Угол (градусы)')
ax1.set_title('Изменение углов двойного маятника (RK4)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t, np.degrees(sol_rk45[:, 0]), 'r-', label='φ1 (первый маятник)')
ax2.plot(t, np.degrees(sol_rk45[:, 1]), 'b-', label='φ2 (второй маятник)')
ax2.set_xlabel('Время (с)')
ax2.set_ylabel('Угол (градусы)')
ax2.set_title('Изменение углов двойного маятника (RK45)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

def get_cartesian_coordinates(phi1, phi2):
    x1 = L1 * np.sin(phi1)
    y1 = -L1 * np.cos(phi1)
    x2 = x1 + L2 * np.sin(phi2)
    y2 = y1 - L2 * np.cos(phi2)
    return x1, y1, x2, y2

fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
ax.set_xlim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
ax.set_ylim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('Движение двойного маятника')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)
ax.plot(0, 0, 'ko', markersize=6)

line, = ax.plot([], [], 'o-', lw=2, markersize=8)
trail1, = ax.plot([], [], 'r-', alpha=0.5, linewidth=1)
trail2, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

trail1_x, trail1_y = [], []
trail2_x, trail2_y = [], []

def init_anim():
    line.set_data([], [])
    trail1.set_data([], [])
    trail2.set_data([], [])
    time_text.set_text('')
    return line, trail1, trail2, time_text

def animate(i):
    x1, y1, x2, y2 = get_cartesian_coordinates(phi1_arr[i], phi2_arr[i])
    
    line.set_data([0, x1, x2], [0, y1, y2])
    
    trail1_x.append(x1)
    trail1_y.append(y1)
    if len(trail1_x) > 100:
        trail1_x.pop(0)
        trail1_y.pop(0)
    trail1.set_data(trail1_x, trail1_y)
    
    trail2_x.append(x2)
    trail2_y.append(y2)
    if len(trail2_x) > 100:
        trail2_x.pop(0)
        trail2_y.pop(0)
    trail2.set_data(trail2_x, trail2_y)
    
    time_text.set_text(f'Время = {t[i]:.2f} с')
    
    return line, trail1, trail2, time_text

ani = anim.FuncAnimation(fig, animate, frames=nsteps, init_func=init_anim,
                        interval=dt*1000, blit=True, repeat=True, repeat_delay=1000)

plt.show()