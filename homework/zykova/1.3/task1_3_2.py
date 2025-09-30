import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PHYSICS
# единица длины = 1 а е
# единица времени = 1 год
# единица массы = масса Земли

a_Earth = 1.0 # большая полуось в а.е.
e_Earth = 0.017 # эксцентриситет орбиты
T_Earth = 1 # орбитальный период в годах
m_Earth = 1

a_Mars = 1.524
e_Mars = 0.094
T_Mars = 1.881
m_Mars = 0.107

a_Jupiter = 5.204
e_Jupiter = 0.049
T_Jupiter = 11.862
m_Jupiter = 317.8

m_Sun = 332946

G = 4 * np.pi**2 / m_Sun

# гравитационный параметр mu = 4 * pi**2 * a**3 / T**2
def GM(a, T):
    return 4 * np.pi**2 * a**3 / T**2 # G * M_sun

def d(s, a, T):
    x, y, vx, vy = s
    r = math.sqrt(x**2 + y**2)
    return np.array([vx, vy, - GM(a, T) * x / r**3, - GM(a, T) * y / r**3])

def init_cond(a, e, T):
    x0, y0 = (1-e)*a, 0
    vx0, vy0 = 0, math.sqrt(GM(a, T) * (1+e) / (a * (1-e)))
    return x0, y0, vx0, vy0

def analytic_solution(a, e):
    theta = np.linspace(0, 2*np.pi, 1000)
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def explicit_euler_step(s, dt, a, T):
    d1 = dt * d(s, a, T)
    return s + d1

def rk4_step(s, dt, a, T):
    d1 = dt * d(s, a, T)
    d2 = dt * d(s + d1 / 2, a, T)
    d3 = dt * d(s + d2 / 2, a, T)
    d4 = dt * d(s + d3, a, T)
    return s + (d1 + 2 * d2 + 2 * d3 + d4) / 6

def timer(T, N):
    t_begin = 0
    t_end = T * 3 
    time = np.linspace(t_begin, t_end, N+1)
    dt = time[1]-time[0]
    return time, dt

def calculate_energy_and_momentum(x, y, vx, vy, a, T):
    r = np.sqrt(x**2 + y**2)
    # орбитальная скорость в квадрате
    v_square = vx**2 + vy**2
    kinetic_energy = 0.5 * v_square
    potential_energy = - GM(a, T) / r
    total_energy = kinetic_energy + potential_energy
    angular_momentum = x * vy - y * vx
    return total_energy, angular_momentum 

def main(a, e, T, N):

    x0, y0, vx0, vy0 = init_cond(a, e, T)
    x_an, y_an = analytic_solution(a, e)
    time, dt = timer(T, N)

    s_Euler = np.zeros((N + 1, 4), dtype=np.float32)
    s_Euler[0] = np.array([x0, y0, vx0, vy0])
    s_RK4 = np.zeros((N + 1, 4), dtype=np.float32)
    s_RK4[0] = np.array([x0, y0, vx0, vy0])

    for i in range(1, N + 1):
        s_Euler[i] = explicit_euler_step(s_Euler[i - 1], dt, a, T)
        s_RK4[i] = rk4_step(s_RK4[i - 1], dt, a, T)

    x_Euler = s_Euler[:, 0]
    y_Euler = s_Euler[:, 1]
    vx_Euler = s_Euler[:, 2]
    vy_Euler = s_Euler[:, 3]

    x_RK4 = s_RK4[:, 0]
    y_RK4 = s_RK4[:, 1]
    vx_RK4 = s_RK4[:, 2]
    vy_RK4 = s_RK4[:, 3]

    E_euler, L_euler = calculate_energy_and_momentum(x_Euler, y_Euler, vx_Euler, vy_Euler, a, T)
    E_rk4, L_rk4 = calculate_energy_and_momentum(x_RK4, y_RK4, vx_RK4, vy_RK4, a, T)

    E_an = 0.5 * (vx0**2 + vy0**2) - GM(a, T) / x0
    L_an = x0 * vy0 - y0 * vx0
    return x_an, y_an, x_Euler, y_Euler, x_RK4, y_RK4, E_euler, E_rk4, L_euler, L_rk4, E_an, L_an, time

N_Earth = 10000
N_Mars = 10000
N_Jupiter = 10000

x_an_E, y_an_E, x_Euler_E, y_Euler_E, x_RK4_E, y_RK4_E, E_euler_E, E_rk4_E, L_euler_E, L_rk4_E, E_an_E, L_an_E, time_E = main(a_Earth, e_Earth, T_Earth, N_Earth)
x_an_M, y_an_M, x_Euler_M, y_Euler_M, x_RK4_M, y_RK4_M, E_euler_M, E_rk4_M, L_euler_M, L_rk4_M, E_an_M, L_an_M, time_M = main(a_Mars, e_Mars, T_Mars, N_Mars)
x_an_J, y_an_J, x_Euler_J, y_Euler_J, x_RK4_J, y_RK4_J, E_euler_J, E_rk4_J, L_euler_J, L_rk4_J, E_an_J, L_an_J, time_J = main(a_Jupiter, e_Jupiter, T_Jupiter, N_Jupiter)

def analytic_orbit(a, e, num_points=1000):
    theta = np.linspace(0, 2*np.pi, num_points)
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def gravitational_force(m1, m2, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    r_sq = dx**2 + dy**2
    r = np.sqrt(r_sq)
    if r < 1e-10: 
        return 0, 0
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

def init_cond_3body(a, e, m_star):
    mu = G * m_star
    r_peri = a * (1 - e)
    v_peri = np.sqrt(mu * (1 + e) / (a * (1 - e)))
    return r_peri, 0, 0, v_peri

def rk4_step_3N(s, dt, m1, m2, m3):
    k1 = dt * d_3N(s, m1, m2, m3)
    k2 = dt * d_3N(s + k1/2, m1, m2, m3)
    k3 = dt * d_3N(s + k2/2, m1, m2, m3)
    k4 = dt * d_3N(s + k3, m1, m2, m3)
    return s + (k1 + 2*k2 + 2*k3 + k4) / 6

def N3_task(a1, a2, e1, e2, m1, m2, m3, duration_years, N):
    x1, y1, vx1, vy1 = init_cond_3body(a1, e1, m3)
    x2, y2, vx2, vy2 = init_cond_3body(a2, e2, m3)
    x3, y3, vx3, vy3 = 0, 0, 0, 0
    time = np.linspace(0, duration_years, N+1)
    dt = time[1] - time[0]
    s = np.zeros((N+1, 12))
    s[0] = np.array([x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3])
    for i in range(1, N+1):
        s[i] = rk4_step_3N(s[i-1], dt, m1, m2, m3)
    return s, time

duration_years = 20
N = 5000

s1, time = N3_task(a_Earth, a_Jupiter, e_Earth, e_Jupiter, m_Earth, m_Jupiter, m_Sun, duration_years, N)
s2, time = N3_task(a_Mars, a_Jupiter, e_Mars, e_Jupiter, m_Mars, m_Jupiter, m_Sun, duration_years, N)

x_E = s1[:, 0] - s1[:, 4]   # Земля x относительно Солнца
y_E = s1[:, 1] - s1[:, 5]   # Земля y относительно Солнца
x_J = s1[:, 2] - s1[:, 4]   # Юпитер x относительно Солнца
y_J = s1[:, 3] - s1[:, 5]   # Юпитер y относительно Солнца
x_M = s2[:, 0] - s2[:, 4]   # Марс x относительно Солнца
y_M = s2[:, 1] - s2[:, 5]   # Марс y относительно Солнца

x_E_analytic, y_E_analytic = analytic_orbit(a_Earth, e_Earth)
x_M_analytic, y_M_analytic = analytic_orbit(a_Mars, e_Mars)
x_J_analytic, y_J_analytic = analytic_orbit(a_Jupiter, e_Jupiter)

fig = plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(x_an_E, y_an_E, 'k--', label='Аналитическая орбита')
plt.plot(x_Euler_E, y_Euler_E, label='Метод Эйлера')
plt.plot(x_RK4_E, y_RK4_E, label='Метод RK4')
plt.scatter([0], [0], c='yellow', s=1000)
plt.xlabel('x (а.е.)')
plt.ylabel('y (а.е.)')
plt.title('Земля')
plt.legend(loc='upper left', frameon=False, fontsize=7)
plt.xlim(-1.05, 1.05)
plt.ylim(-1.05, 1.05)

plt.subplot(2, 2, 2)
plt.plot(x_an_M, y_an_M, 'k--', label='Аналитическая орбита')
plt.plot(x_Euler_M, y_Euler_M, label='Метод Эйлера')
plt.plot(x_RK4_M, y_RK4_M, label='Метод RK4')
plt.scatter([0], [0], c='yellow', s=1000)
plt.xlabel('x (а.е.)')
plt.ylabel('y (а.е.)')
plt.title('Марс')
plt.legend(loc='upper left', frameon=False, fontsize=7)
plt.xlim(-1.7, 1.7)
plt.ylim(-1.7, 1.7)

plt.subplot(2, 2, 3)
plt.plot(x_an_J, y_an_J, 'k--', label='Аналитическая орбита')
plt.plot(x_Euler_J, y_Euler_J, label='Метод Эйлера')
plt.plot(x_RK4_J, y_RK4_J, label='Метод RK4')
plt.scatter([0], [0], c='yellow', s=1000) 
plt.xlabel('x (а.е.)')
plt.ylabel('y (а.е.)')
plt.title('Юпитер')
plt.legend(loc='upper left', frameon=False, fontsize=7)
plt.xlim(-5.5, 5.5)
plt.ylim(-5.5, 5.5)

plt.subplot(2, 2, 4)
plt.plot(x_an_E, y_an_E, 'k--', label='Земля')
plt.plot(x_an_M, y_an_M, label='Марс')
plt.plot(x_an_J, y_an_J, label='Юпитер')
plt.scatter([0], [0], c='yellow', s=100)
plt.xlabel('x (а.е.)')
plt.ylabel('y (а.е.)')
plt.title('Аналитические орбиты')
plt.legend(loc='upper left', frameon=False, fontsize=7)
plt.xlim(-5.5, 5.5)
plt.ylim(-5.5, 5.5)

plt.suptitle('Визуализация орбит для разных эксцентриситетов', 
             fontsize=16, y=0.98)

plt.tight_layout()

plt.show()

def create_three_planets_animation():

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_xlabel('x (а.е.)')
    ax.set_ylabel('y (а.е.)')
    ax.set_title('Анимация движения планет. Аналитическое решение')
    ax.scatter([0], [0], c='yellow', s=200, label='Солнце')

    ax.plot(x_an_E, y_an_E, 'b--', alpha=0.5, linewidth=0.8, label='Орбита Земли')
    ax.plot(x_an_M, y_an_M, 'r--', alpha=0.5, linewidth=0.8, label='Орбита Марса')
    ax.plot(x_an_J, y_an_J, 'g--', alpha=0.5, linewidth=0.8, label='Орбита Юпитера')
    
    earth_point, = ax.plot([], [], 'bo', markersize=6, label='Земля')
    mars_point, = ax.plot([], [], 'ro', markersize=6, label='Марс')
    jupiter_point, = ax.plot([], [], 'go', markersize=10, label='Юпитер')
    
    earth_trail, = ax.plot([], [], 'b-', alpha=0.3, linewidth=1)
    mars_trail, = ax.plot([], [], 'r-', alpha=0.3, linewidth=1)
    jupiter_trail, = ax.plot([], [], 'g-', alpha=0.3, linewidth=1)
    
    time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.legend(loc='upper right', frameon=True, fontsize=8)
    
    def animate(frame):

        total_simulation_time = 6 * T_Earth
        
        current_time = (frame / frames) * total_simulation_time
        
        angle_earth = (2 * np.pi / T_Earth) * current_time
        angle_mars = (2 * np.pi / T_Mars) * current_time
        angle_jupiter = (2 * np.pi / T_Jupiter) * current_time
        
        theta = np.linspace(0, 2*np.pi, len(x_an_E))
        
        idx_earth = np.argmin(np.abs((theta - angle_earth) % (2*np.pi)))
        idx_mars = np.argmin(np.abs((theta - angle_mars) % (2*np.pi)))
        idx_jupiter = np.argmin(np.abs((theta - angle_jupiter) % (2*np.pi)))
        
        earth_point.set_data([x_an_E[idx_earth]], [y_an_E[idx_earth]])
        mars_point.set_data([x_an_M[idx_mars]], [y_an_M[idx_mars]])
        jupiter_point.set_data([x_an_J[idx_jupiter]], [y_an_J[idx_jupiter]])
        
        trail_length = 50
        start_e = max(0, idx_earth - trail_length)
        start_m = max(0, idx_mars - trail_length)
        start_j = max(0, idx_jupiter - trail_length)
        
        earth_trail.set_data(x_an_E[start_e:idx_earth+1], y_an_E[start_e:idx_earth+1])
        mars_trail.set_data(x_an_M[start_m:idx_mars+1], y_an_M[start_m:idx_mars+1])
        jupiter_trail.set_data(x_an_J[start_j:idx_jupiter+1], y_an_J[start_j:idx_jupiter+1])
        
        time_text.set_text(f'Время: {current_time:.2f} лет\n'
                          f'Земля: {angle_earth/np.pi:.2f}π рад\n'
                          f'Марс: {angle_mars/np.pi:.2f}π рад\n'
                          f'Юпитер: {angle_jupiter/np.pi:.2f}π рад')
        
        return (earth_point, mars_point, jupiter_point, 
                earth_trail, mars_trail, jupiter_trail, time_text)
    
    frames = 300 
    
    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        frames=frames,
        interval=50,  
        blit=True,
        repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    
    return ani

create_three_planets_animation()

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.set_xlabel('X (а.е.)')
ax.set_ylabel('Y (а.е.)')
ax.set_title('Движение Земли, Марса с учетом Юпитера вокруг Солнца\n(RK4: сплошные линии, аналитика: пунктир)')

earth_analytic, = ax.plot(x_E_analytic, y_E_analytic, 'b--', alpha=0.7, linewidth=1, label='Аналит. орбита Земли')
mars_analytic, = ax.plot(x_M_analytic, y_M_analytic, 'r--', alpha=0.7, linewidth=1, label='Аналит. орбита Марса')
jupiter_analytic, = ax.plot(x_J_analytic, y_J_analytic, 'g--', alpha=0.7, linewidth=1, label='Аналит. орбита Юпитера')

sun, = ax.plot([0], [0], 'yo', markersize=15, label='Солнце')
earth_trail, = ax.plot([], [], 'b-', alpha=0.8, linewidth=2, label='Земля, RK4')
mars_trail, = ax.plot([], [], 'r-', alpha=0.8, linewidth=2, label='Марс, RK4')
jupiter_trail, = ax.plot([], [], 'g-', alpha=0.8, linewidth=2, label='Юпитер, RK4')
earth_point, = ax.plot([], [], 'bo', markersize=8, label='Земля')
mars_point, = ax.plot([], [], 'ro', markersize=8, label='Марс')
jupiter_point, = ax.plot([], [], 'go', markersize=10, label='Юпитер')

ax.legend(loc='upper right', fontsize=10)
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

def init():
    earth_trail.set_data([], [])
    mars_trail.set_data([], [])
    jupiter_trail.set_data([], [])
    earth_point.set_data([], [])
    mars_point.set_data([], [])
    jupiter_point.set_data([], [])
    time_text.set_text('')
    return earth_trail, mars_trail, jupiter_trail, earth_point, mars_point, jupiter_point, time_text

def animate(i):
    start_idx = max(0, i - 200)
    earth_trail.set_data(x_E[start_idx:i+1], y_E[start_idx:i+1])
    earth_point.set_data([x_E[i]], [y_E[i]])
    mars_trail.set_data(x_M[start_idx:i+1], y_M[start_idx:i+1])
    mars_point.set_data([x_M[i]], [y_M[i]])
    jupiter_trail.set_data(x_J[start_idx:i+1], y_J[start_idx:i+1])
    jupiter_point.set_data([x_J[i]], [y_J[i]])
    
    def find_closest_analytic_point(x_num, y_num, x_an, y_an):
        distances = np.sqrt((x_an - x_num)**2 + (y_an - y_num)**2)
        min_idx = np.argmin(distances)
        return distances[min_idx]
    
    dev_E = find_closest_analytic_point(x_E[i], y_E[i], x_E_analytic, y_E_analytic)
    dev_M = find_closest_analytic_point(x_M[i], y_M[i], x_M_analytic, y_M_analytic)
    
    years = time[i]
    time_text.set_text(f'Время: {years:.1f} лет\n'
                      f'Отклонение Земли: {dev_E:.4f} а.е.\n'
                      f'Отклонение Марса: {dev_M:.4f} а.е.')
    
    return earth_trail, mars_trail, jupiter_trail, earth_point, mars_point, jupiter_point, time_text

anim = animation.FuncAnimation(fig, animate, frames=len(time), init_func=init,
                              interval=20, blit=True)
plt.tight_layout()
plt.show()

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(x_E_analytic, y_E_analytic, 'b--', alpha=0.5, label='Аналит. Земля')
ax1.plot(x_E, y_E, 'b-', alpha=0.8, label='RK4 Земля')
ax1.plot(x_M_analytic, y_M_analytic, 'r--', alpha=0.5, label='Аналит. Марс')
ax1.plot(x_M, y_M, 'r-', alpha=0.8, label='RK4 Марс')
ax1.plot(x_J_analytic, y_J_analytic, 'g--', alpha=0.5, label='Аналит. Юпитер')
ax1.plot(x_J, y_J, 'g-', alpha=0.8, label='RK4 Юпитер')
ax1.plot(0, 0, 'yo', markersize=10, label='Солнце')
ax1.set_xlim(-7, 7)
ax1.set_ylim(-7, 7)
ax1.set_xlabel('X (а.е.)')
ax1.set_ylabel('Y (а.е.)')
ax1.set_title('Сравнение численных и аналитических орбит')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

deviations_E = []
deviations_M = []

def true_anomaly(x, y):
    return np.arctan2(y, x)

for i in range(0, len(x_E), 100):
    theta_E = true_anomaly(x_E[i], y_E[i])
    theta_M = true_anomaly(x_M[i], y_M[i])
    r_E_analytic = a_Earth * (1 - e_Earth**2) / (1 + e_Earth * np.cos(theta_E))
    r_M_analytic = a_Mars * (1 - e_Mars**2) / (1 + e_Mars * np.cos(theta_M))
    r_E_current = np.sqrt(x_E[i]**2 + y_E[i]**2)
    r_M_current = np.sqrt(x_M[i]**2 + y_M[i]**2)
    deviations_E.append(abs(r_E_current - r_E_analytic))
    deviations_M.append(abs(r_M_current - r_M_analytic))

time_sample = time[::100]

ax2.plot(time_sample, deviations_E, 'b-', label='Отклонение Земли')
ax2.plot(time_sample, deviations_M, 'r-', label='Отклонение Марса')
ax2.set_xlabel('Время (годы)')
ax2.set_ylabel('Отклонение (а.е.)')
ax2.set_title('Отклонение от аналитических орбит')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax2.text(0.5, -0.15, f'Максимальное отклонение Земли: {max(deviations_E):.6f} а.е. | '
                     f'Максимальное отклонение Марса: {max(deviations_M):.6f} а.е.', 
         transform=ax2.transAxes, fontsize=10,
         horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.show()

