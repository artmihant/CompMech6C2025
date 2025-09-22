import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim

L1 = 1.0
L2 = 1.0
m1 = 1.0
m2 = 1.0 
g = 9.81

phi1_0 = math.radians(5.0)  
phi2_0 = math.radians(0.0)
psi1_0 = 0.0                
psi2_0 = 0.0

time = 10.0
nsteps = 1000
dt = time / nsteps
t = np.linspace(0, time, nsteps + 1)

phi1_arr = np.zeros(nsteps + 1)
phi2_arr = np.zeros(nsteps + 1)
psi1_arr = np.zeros(nsteps + 1)
psi2_arr = np.zeros(nsteps + 1)

phi1_arr[0], phi2_arr[0] = phi1_0, phi2_0
psi1_arr[0], psi2_arr[0] = psi1_0, psi2_0

def equations_of_motion(phi1, phi2, psi1, psi2):
    delta = phi2 - phi1
    
    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)
    
    denom = m1 + m2 * sin_delta**2
    
    psi1_dot = (m2 * g * phi2 - (m1 + m2) * g * phi1) / (L1 * m1)
    
    psi2_dot = (m1 + m2) * (- g * phi2 + g * phi1) / (L2 * m1)
    
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

def get_cartesian_coordinates(phi1, phi2):
    x1 = L1 * math.sin(phi1)
    y1 = -L1 * math.cos(phi1)
    x2 = x1 + L2 * math.sin(phi2)
    y2 = y1 - L2 * math.cos(phi2)
    return x1, y1, x2, y2

phi1, phi2, psi1, psi2 = phi1_0, phi2_0, psi1_0, psi2_0

for i in range(nsteps):
    phi1, phi2, psi1, psi2 = rk4_step(phi1, phi2, psi1, psi2, dt)
    phi1_arr[i + 1] = phi1
    phi2_arr[i + 1] = phi2
    psi1_arr[i + 1] = psi1
    psi2_arr[i + 1] = psi2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(t, np.degrees(phi1_arr), 'r-', label='φ1 (первый маятник)')
ax1.plot(t, np.degrees(phi2_arr), 'b-', label='φ2 (второй маятник)')
ax1.set_xlabel('Время (с)')
ax1.set_ylabel('Угол (градусы)')
ax1.set_title('Изменение углов двойного маятника')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t, psi1_arr, 'r--', label='ω1 (первый маятник)')
ax2.plot(t, psi2_arr, 'b--', label='ω2 (второй маятник)')
ax2.set_xlabel('Время (с)')
ax2.set_ylabel('Угловая скорость (рад/с)')
ax2.set_title('Изменение угловых скоростей')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
ax.set_xlim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
ax.set_ylim(-(L1 + L2) * 1.2, (L1 + L2) * 0.2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('Движение двойного маятника')

line, = ax.plot([], [], 'o-', lw=2)
trail, = ax.plot([], [], 'g-', alpha=0.5)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

trail_x, trail_y = [], []

def init_anim():
    line.set_data([], [])
    trail.set_data([], [])
    time_text.set_text('')
    return line, trail, time_text

def animate(i):
    x1, y1, x2, y2 = get_cartesian_coordinates(phi1_arr[i], phi2_arr[i])
    
    line.set_data([0, x1, x2], [0, y1, y2])
    
    trail_x.append(x2)
    trail_y.append(y2)
    if len(trail_x) > 100:
        trail_x.pop(0)
        trail_y.pop(0)
    trail.set_data(trail_x, trail_y)
    
    time_text.set_text(f'Время = {t[i]:.2f} с')
    
    return line, trail, time_text

ani = anim.FuncAnimation(fig, animate, frames=nsteps, init_func=init_anim,
                        interval=dt*1000, blit=True)

plt.show()