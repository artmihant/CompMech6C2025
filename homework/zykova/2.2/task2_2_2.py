import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = 1.0 

# Восьмерка
m1_1, m2_1, m3_1 = 1.0, 1.0, 1.0
x1_1, y1_1 = -0.97000436, 0.24308753
x2_1, y2_1 = 0.97000436, -0.24308753
x3_1, y3_1 = 0.0, 0.0
vx1_1, vy1_1 = 0.4662036850, 0.4323657300
vx2_1, vy2_1 = 0.4662036850, 0.4323657300
vx3_1, vy3_1 = -0.93240737, -0.86473146

# Решение Лагранжа
m1_2, m2_2, m3_2 = 1.0, 1.0, 1.0
side = 2.0
x1_2, y1_2 = 0.0, 0.0
x2_2, y2_2 = side, 0.0
x3_2, y3_2 = side/2, side * math.sqrt(3)/2

cm_x = (m1_2*x1_2 + m2_2*x2_2 + m3_2*x3_2) / (m1_2 + m2_2 + m3_2)
cm_y = (m1_2*y1_2 + m2_2*y2_2 + m3_2*y3_2) / (m1_2 + m2_2 + m3_2)

x1_2 -= cm_x; y1_2 -= cm_y
x2_2 -= cm_x; y2_2 -= cm_y
x3_2 -= cm_x; y3_2 -= cm_y

omega = math.sqrt(G * 3.0 / side**3)
vx1_2, vy1_2 = -omega * y1_2, omega * x1_2
vx2_2, vy2_2 = -omega * y2_2, omega * x2_2
vx3_2, vy3_2 = -omega * y3_2, omega * x3_2

# Две тяжелые звезды и одна легкая планета
m1_3, m2_3, m3_3 = 1.0, 1.0, 0.01
distance = 1.0
x1_3, y1_3 = -distance/2, 0.0
x2_3, y2_3 = distance/2, 0.0
x3_3, y3_3 = 0.0, 2.5

total_mass = m1_3 + m2_3 + m3_3
cm_x3 = (m1_3*x1_3 + m2_3*x2_3 + m3_3*x3_3) / total_mass
cm_y3 = (m1_3*y1_3 + m2_3*y2_3 + m3_3*y3_3) / total_mass
x1_3 -= cm_x3; y1_3 -= cm_y3
x2_3 -= cm_x3; y2_3 -= cm_y3
x3_3 -= cm_x3; y3_3 -= cm_y3

v = math.sqrt(G * (m1_3 + m2_3) / (4 * distance))
vx1_3, vy1_3 = 0.0, v
vx2_3, vy2_3 = 0.0, -v
dx = x3_3 - (m1_3*x1_3 + m2_3*x2_3) / (m1_3 + m2_3)
dy = y3_3 - (m1_3*y1_3 + m2_3*y2_3) / (m1_3 + m2_3)
distance = math.sqrt(dx**2 + dy**2)
v_orbital = math.sqrt(G * (m1_3 + m2_3) / distance)
vx3_3 = -v_orbital * dy / distance
vy3_3 = v_orbital * dx / distance

def gravitational_force(m1, m2, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    r_sq = dx**2 + dy**2
    if r_sq < 1e-12: 
        return 0, 0
    r = np.sqrt(r_sq)
    force_magnitude = G * m1 * m2 / r_sq
    fx = force_magnitude * dx / r
    fy = force_magnitude * dy / r
    return fx, fy

def d_3N(s, m1, m2, m3):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = s
    f12x, f12y = gravitational_force(m1, m2, x1, y1, x2, y2)
    f13x, f13y = gravitational_force(m1, m3, x1, y1, x3, y3)
    f23x, f23y = gravitational_force(m2, m3, x2, y2, x3, y3)
    ax1 = (f12x + f13x) / m1
    ay1 = (f12y + f13y) / m1
    ax2 = (-f12x + f23x) / m2
    ay2 = (-f12y + f23y) / m2
    ax3 = (-f13x - f23x) / m3
    ay3 = (-f13y - f23y) / m3
    return np.array([vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3])

def rk4_step_3N(s, dt, m1, m2, m3):
    k1 = dt * d_3N(s, m1, m2, m3)
    k2 = dt * d_3N(s + k1/2, m1, m2, m3)
    k3 = dt * d_3N(s + k2/2, m1, m2, m3)
    k4 = dt * d_3N(s + k3, m1, m2, m3)
    return s + (k1 + 2*k2 + 2*k3 + k4) / 6

def estimate_stable_timestep(s, m1, m2, m3, safety_factor=0.01):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = s
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    r_min = min(r12, r13, r23)
    if r_min < 1e-8:
        return 1e-6
    total_mass = m1 + m2 + m3
    orbital_period_estimate = 2 * np.pi * np.sqrt(r_min**3 / (G * total_mass))
    dt_stable = safety_factor * orbital_period_estimate
    return min(dt_stable, 0.1)

def total_energy(s, m1, m2, m3):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = s
    kinetic = 0.5 * (m1*(vx1**2+vy1**2) + m2*(vx2**2+vy2**2) + m3*(vx3**2+vy3**2))
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    potential = -G * (m1*m2/r12 + m1*m3/r13 + m2*m3/r23)
    return kinetic + potential

def adaptive_N3_task(m1, m2, m3, x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3, duration, max_points=10000):
    s_current = np.array([x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3])
    dt = estimate_stable_timestep(s_current, m1, m2, m3)
    results = [s_current.copy()]
    times = [0.0]
    energies = [total_energy(s_current, m1, m2, m3)]
    t_current = 0.0
    point_count = 1
    while t_current < duration and point_count < max_points:
        s_prev = s_current.copy()
        s_try = rk4_step_3N(s_current, dt, m1, m2, m3)
        r12 = np.sqrt((s_try[2]-s_try[0])**2 + (s_try[3]-s_try[1])**2)
        r13 = np.sqrt((s_try[4]-s_try[0])**2 + (s_try[5]-s_try[1])**2)
        r23 = np.sqrt((s_try[4]-s_try[2])**2 + (s_try[5]-s_try[3])**2)
        min_distance = min(r12, r13, r23)
        if min_distance < 1e-10 or np.any(np.isnan(s_try)) or np.any(np.isinf(s_try)):
            dt *= 0.5
            continue
        s_current = s_try
        t_current += dt
        if point_count % 10 == 0 or t_current - times[-1] > duration/100:
            results.append(s_current.copy())
            times.append(t_current)
            energies.append(total_energy(s_current, m1, m2, m3))
            point_count += 1
        new_dt = estimate_stable_timestep(s_current, m1, m2, m3)
        dt = min(new_dt, duration - t_current)
    return np.array(results), np.array(times), np.array(energies)

duration = 25
max_points = 15000

s1, time1, energy1 = adaptive_N3_task(m1_1, m2_1, m3_1, x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, vx1_1, vy1_1, vx2_1, vy2_1, vx3_1, vy3_1, duration, max_points)
s2, time2, energy2 = adaptive_N3_task(m1_2, m2_2, m3_2, x1_2, y1_2, x2_2, y2_2, x3_2, y3_2, vx1_2, vy1_2, vx2_2, vy2_2, vx3_2, vy3_2, duration, max_points)
s3, time3, energy3 = adaptive_N3_task(m1_3, m2_3, m3_3, x1_3, y1_3, x2_3, y2_3, x3_3, y3_3, vx1_3, vy1_3, vx2_3, vy2_3, vx3_3, vy3_3, duration, max_points)

x1_1_traj, y1_1_traj = s1[:, 0], s1[:, 1]
x2_1_traj, y2_1_traj = s1[:, 2], s1[:, 3]
x3_1_traj, y3_1_traj = s1[:, 4], s1[:, 5]

x1_2_traj, y1_2_traj = s2[:, 0], s2[:, 1]
x2_2_traj, y2_2_traj = s2[:, 2], s2[:, 3]
x3_2_traj, y3_2_traj = s2[:, 4], s2[:, 5]

x1_3_traj, y1_3_traj = s3[:, 0], s3[:, 1]
x2_3_traj, y2_3_traj = s3[:, 2], s3[:, 3]
x3_3_traj, y3_3_traj = s3[:, 4], s3[:, 5]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax1, ax2, ax3 = axes[0], axes[1], axes[2]

ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2); ax1.set_title('Фигура 8'); ax1.grid(True, alpha=0.3)
ax2.set_xlim(-2, 2); ax2.set_ylim(-2, 2); ax2.set_title('Треугольник Лагранжа'); ax2.grid(True, alpha=0.3)
ax3.set_xlim(-3, 3); ax3.set_ylim(-3, 3); ax3.set_title('Двойная система'); ax3.grid(True, alpha=0.3)

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_aspect('equal')

def create_bodies_trajectories(ax, color1='red', color2='blue', color3='green'):
    body1, = ax.plot([], [], 'o', color=color1, markersize=8, label='Тело 1')
    body2, = ax.plot([], [], 'o', color=color2, markersize=8, label='Тело 2') 
    body3, = ax.plot([], [], 'o', color=color3, markersize=8, label='Тело 3')
    traj1, = ax.plot([], [], '-', color=color1, alpha=0.4, linewidth=1.5)
    traj2, = ax.plot([], [], '-', color=color2, alpha=0.4, linewidth=1.5)
    traj3, = ax.plot([], [], '-', color=color3, alpha=0.4, linewidth=1.5)
    return body1, body2, body3, traj1, traj2, traj3

bodies1 = create_bodies_trajectories(ax1)
bodies2 = create_bodies_trajectories(ax2)  
bodies3 = create_bodies_trajectories(ax3)

for ax in [ax1, ax2, ax3]:
    ax.legend(loc='upper right')

def init_combined():
    elements = []
    for bodies in [bodies1, bodies2, bodies3]:
        for element in bodies:
            element.set_data([], [])
            elements.append(element)
    return tuple(elements)

def update_combined(frame):
    elements = []
    
    frame1 = min(frame, len(x1_1_traj)-1)
    frame2 = min(frame, len(x1_2_traj)-1) 
    frame3 = min(frame, len(x1_3_traj)-1)
    
    for i, (bodies, x1, y1, x2, y2, x3, y3, frame_idx) in enumerate([
        (bodies1, x1_1_traj, y1_1_traj, x2_1_traj, y2_1_traj, x3_1_traj, y3_1_traj, frame1),
        (bodies2, x1_2_traj, y1_2_traj, x2_2_traj, y2_2_traj, x3_2_traj, y3_2_traj, frame2),
        (bodies3, x1_3_traj, y1_3_traj, x2_3_traj, y2_3_traj, x3_3_traj, y3_3_traj, frame3)
    ]):
        body1, body2, body3, traj1, traj2, traj3 = bodies
        body1.set_data([x1[frame_idx]], [y1[frame_idx]])
        body2.set_data([x2[frame_idx]], [y2[frame_idx]]) 
        body3.set_data([x3[frame_idx]], [y3[frame_idx]])
        traj1.set_data(x1[:frame_idx+1], y1[:frame_idx+1])
        traj2.set_data(x2[:frame_idx+1], y2[:frame_idx+1])
        traj3.set_data(x3[:frame_idx+1], y3[:frame_idx+1])
        elements.extend([body1, body2, body3, traj1, traj2, traj3])
    
    return tuple(elements)

max_frames = min(len(x1_1_traj), len(x1_2_traj), len(x1_3_traj))
step = max(1, max_frames // 200)
ani_frames = np.arange(0, max_frames, step)
ani_combined = animation.FuncAnimation(fig, update_combined, frames=ani_frames, init_func=init_combined, blit=True, interval=50)

plt.tight_layout()
plt.show()