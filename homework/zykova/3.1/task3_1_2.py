import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = 66.743e-10 # м^3 / (кг * с^2) 

mE = 5.9736e24
mL = 7.36e22
m = 1000

# Начальные условия
xE, yE, vxE, vyE = 0, 0, 0, 0
xL, yL = 384400e3, 0
vxL, vyL = 0, np.sqrt(G * mE / xL)  # Орбитальная скорость Луны

# Радиус Луны 
# Область, попадание в которую считаем решением
moon_radius = 1737000  # м

def gravitational_force(m1, m2, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    r_sq = dx**2 + dy**2
    if r_sq < 1e-20: 
        return 0, 0
    r = np.sqrt(r_sq)
    force_magnitude = G * m1 * m2 / r_sq
    fx = force_magnitude * dx / r
    fy = force_magnitude * dy / r
    return fx, fy

def d_3N(s, m1, m2, m3):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = s
    f12x, f12y = gravitational_force(m1, m2, x1, y1, x2, y2)  # Земля-Луна
    f13x, f13y = gravitational_force(m1, m3, x1, y1, x3, y3)  # Земля-Снаряд
    f23x, f23y = gravitational_force(m2, m3, x2, y2, x3, y3)  # Луна-Снаряд
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

def simulate_trajectory(theta, duration=2e6, N=100000):
    # Начальные условия для снаряда (старт с поверхности Земли)
    x = 6371e3 
    y = 0
    v0 = np.sqrt(2 * G * mE / x)  # Вторая космическая скорость
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    
    time = np.linspace(0, duration, N+1)
    dt = time[1] - time[0]
    s = np.zeros((N+1, 12))
    s[0] = np.array([xE, yE, xL, yL, x, y, vxE, vyE, vxL, vyL, vx, vy])
    
    min_distance = float('inf')
    reached_moon = False
    trajectory_data = None
    
    for i in range(1, N+1):
        s[i] = rk4_step_3N(s[i-1], dt, mE, mL, m)
        x3, y3 = s[i, 4], s[i, 5]
        xL_current, yL_current = s[i, 2], s[i, 3]
        distance_to_moon = np.sqrt((x3 - xL_current)**2 + (y3 - yL_current)**2)
        if distance_to_moon < min_distance:
            min_distance = distance_to_moon
        if distance_to_moon <= moon_radius and not reached_moon:
            reached_moon = True
            trajectory_data = s[:i+1]
            break
    if trajectory_data is None:
        trajectory_data = s
    
    return reached_moon, min_distance, trajectory_data, time[:len(trajectory_data)]

def shooting_method(theta_min=0, theta_max=np.pi/2, tolerance=0.001, max_iterations=15):
    print("МЕТОД СТРЕЛЬБЫ (БИСЕКЦИИ)")
    print(f"Диапазон углов: {np.degrees(theta_min):.2f}° - {np.degrees(theta_max):.2f}°")
    best_theta = None
    best_distance = float('inf')
    best_trajectory = None
    for iteration in range(max_iterations):
        print(f"Итерация {iteration+1}: диапазон {np.degrees(theta_min):.2f}° - {np.degrees(theta_max):.2f}°")
        theta_mid = (theta_min + theta_max) / 2
        theta_test = [theta_min, theta_mid, theta_max]
        distances = []
        reached_moons = []
        trajectories = []
        for theta in theta_test:
            reached_moon, min_distance, trajectory, time_data = simulate_trajectory(theta)
            distances.append(min_distance)
            reached_moons.append(reached_moon)
            trajectories.append(trajectory)
            status = "достигнуты" if reached_moon else "не достигнута"
            print(f"θ = {np.degrees(theta):.2f}°: Луна {status}, расстояние: {min_distance/1000:.2f} км")
        if any(reached_moons):
            valid_indices = [i for i in range(3) if reached_moons[i]]
            best_idx = valid_indices[np.argmin([distances[i] for i in valid_indices])]
            best_theta = theta_test[best_idx]
            best_distance = distances[best_idx]
            best_trajectory = trajectories[best_idx]
            print(f"\nНайден угол достижения Луны: θ = {np.degrees(best_theta):.4f}°")
            break
        if distances[0] < distances[2]:
            theta_max = theta_mid
        else:
            theta_min = theta_mid
        range_width = theta_max - theta_min
        if range_width < tolerance:
            best_theta = theta_mid
            best_distance = min(distances)
            print(f"Достигнута требуемая точность.")
            break
        print()
    return best_theta, best_distance, best_trajectory

def newton_method(theta_initial=np.pi/2, tolerance=1e-6, max_iterations=10, h=1e-4):
    print("\nМЕТОД НЬЮТОНА")
    print(f"Начальное приближение: θ = {np.degrees(theta_initial):.2f}°")
    theta = theta_initial
    best_theta = None
    best_distance = float('inf')
    best_trajectory = None
    for iteration in range(max_iterations):
        reached_moon, f_theta, trajectory, time_data = simulate_trajectory(theta)
        if f_theta < best_distance:
            best_distance = f_theta
            best_theta = theta
            best_trajectory = trajectory
        print(f"Итерация {iteration+1}: θ = {np.degrees(theta):.6f}°")
        status = "достигнута" if reached_moon else "не достигнута"
        print(f"Луна {status}, расстояние: {f_theta/1000:.6f} км")
        if reached_moon and f_theta <= moon_radius:
            break
        _, f_theta_plus, _, _ = simulate_trajectory(theta + h)  # f(theta + h)  
        _, f_theta_minus, _, _ = simulate_trajectory(theta - h)   # f(theta - h)
        derivative = (f_theta_plus - f_theta_minus) / (2 * h)
        if abs(derivative) < 1e-10:
            print("  Производная близка к нулю, завершаем итерации")
            break
        theta_new = theta - f_theta / derivative
        if abs(theta_new - theta) < tolerance:
            theta = theta_new
            print(f"\nДостигнута требуемая точность.")
            break
        theta = theta_new
        if theta < 0:
            theta = 0.01
        elif theta > np.pi/2:
            theta = np.pi/2 - 0.01
        print()
    return best_theta, best_distance, best_trajectory

theta_shooting, dist_shooting, traj_shooting = shooting_method()
theta_newton, dist_newton, traj_newton = newton_method()

def create_animation(trajectory_data, theta, method_name):
    x_earth = trajectory_data[:, 0]  
    y_earth = trajectory_data[:, 1]
    x_moon = trajectory_data[:, 2]   
    y_moon = trajectory_data[:, 3]
    x_body = trajectory_data[:, 4]  
    y_body = trajectory_data[:, 5]
    margin = 100000e3  
    x_min = min(np.min(x_earth), np.min(x_moon), np.min(x_body)) - margin
    x_max = max(np.max(x_earth), np.max(x_moon), np.max(x_body)) + margin
    y_min = min(np.min(y_earth), np.min(y_moon), np.min(y_body)) - margin
    y_max = max(np.max(y_earth), np.max(y_moon), np.max(y_body)) + margin
        
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X координата, м')
    ax.set_ylabel('Y координата, м')
    ax.set_title(f'Движение тела от Земли к Луне, {method_name}')
    earth = plt.Circle((0, 0), 6371e3, color='blue', alpha=0.7, label='Земля')
    moon = plt.Circle((x_moon[0], y_moon[0]), 1737e3, color='gray', alpha=0.7, label='Луна')
    body, = ax.plot([], [], 'ro', markersize=4, label='Снаряд')
    body_trail, = ax.plot([], [], 'r-', alpha=0.5, linewidth=1)
    moon_trail, = ax.plot([], [], 'g-', alpha=0.3, linewidth=1)
    ax.add_patch(earth)
    ax.add_patch(moon)
    ax.legend()
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    def init():
        body.set_data([], [])
        body_trail.set_data([], [])
        moon_trail.set_data([], [])
        moon.center = (x_moon[0], y_moon[0])
        info_text.set_text('')
        return body, body_trail, moon_trail, moon, info_text
    def animate(i):
        i = min(i, len(x_body) - 1)
        body.set_data([x_body[i]], [y_body[i]])
        trail_start = max(0, i - 100)
        body_trail.set_data(x_body[trail_start:i+1], y_body[trail_start:i+1])
        moon_trail.set_data(x_moon[:i+1], y_moon[:i+1])
        moon.center = (x_moon[i], y_moon[i])
        distance_to_moon = np.sqrt((x_body[i] - x_moon[i])**2 + (y_body[i] - y_moon[i])**2)
        time_hours = i * 2e6 / 3600 / len(x_body)  
        info_text.set_text(f'Время: {time_hours:.2f} ч\n'
                            f'Расстояние до Луны: {distance_to_moon/1000:.0f} км\n'
                            f'Угол запуска: {np.degrees(theta):.4f}°')
        return body, body_trail, moon_trail, moon, info_text
    frames = len(x_body)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=20, blit=True)
    plt.tight_layout()
    return anim, fig

anim, fig = create_animation(traj_shooting, theta_shooting, method_name="Метод Стрельбы")
plt.show()
anim, fig = create_animation(traj_newton, theta_newton, method_name="Метод Ньютона")
plt.show()

def trajetory_plot(trajectory_data, theta, method_name):
    plt.figure(figsize=(12, 10))
    plt.plot(trajectory_data[:, 4], trajectory_data[:, 5], 'r-', alpha=0.7, linewidth=2, label='Траектория снаряда')
    plt.plot(trajectory_data[:, 2], trajectory_data[:, 3], 'g-', alpha=0.5, linewidth=1, label='Орбита Луны')
    plt.plot(trajectory_data[0, 4], trajectory_data[0, 5], 'go', markersize=8, label='Старт (Земля)')
    plt.plot(trajectory_data[-1, 4], trajectory_data[-1, 5], 'bo', markersize=8, label='Финиш')
    earth_circle = plt.Circle((0, 0), 6371e3, color='blue', alpha=0.3, label='Земля')
    moon_final_x, moon_final_y = trajectory_data[-1, 2], trajectory_data[-1, 3]
    moon_circle = plt.Circle((moon_final_x, moon_final_y), 1737e3, color='gray', alpha=0.3, label='Луна')
    plt.gca().add_patch(earth_circle)
    plt.gca().add_patch(moon_circle)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('X координата, м')
    plt.ylabel('Y координата, м')
    plt.title(f'Полная траектория движения, {method_name}: {np.degrees(theta):.6f}°')
    plt.legend()
    plt.tight_layout()
    plt.show()

trajetory_plot(traj_shooting, theta_shooting, method_name="Метод Стрельбы")
trajetory_plot(traj_newton, theta_newton, method_name="Метод Ньютона")