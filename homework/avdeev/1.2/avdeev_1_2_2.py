import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.81

def simulate_projectile(v0, alpha, g=g, dt=0.002, max_time=30.0):
    vx = v0 * math.cos(alpha)
    vy = v0 * math.sin(alpha)
    x = 0.0
    y = 0.0

    xs = [x]
    ys = [y]

    t = 0.0
    while t < max_time:
        # если уже ушли под землю и это не первый шаг -- прерываем
        if t > 0 and y < 0:
            break

        def deriv(state):
            return np.array([state[1], -g])

        state = np.array([y, vy], dtype=float)
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        y_new, vy_new = float(state_new[0]), float(state_new[1])
        x_new = x + vx * dt

        xs.append(x_new)
        ys.append(y_new)

        x, y, vy = x_new, y_new, vy_new
        t += dt

    # интерполяция пересечения с землёй, если надо
    if ys[-1] < 0 and len(ys) >= 2:
        x1, y1 = xs[-2], ys[-2]
        x2, y2 = xs[-1], ys[-1]
        if y2 == y1:
            x_ground = x2
        else:
            frac = -y1 / (y2 - y1)
            x_ground = x1 + frac * (x2 - x1)
            xs[-1] = x_ground
            ys[-1] = 0.0
    else:
        x_ground = xs[-1]

    return np.array(xs), np.array(ys), float(x_ground)

def analytic_range(v0, alpha, g=g):
    return v0 * v0 * math.sin(2 * alpha) / g

def numeric_range(v0, alpha, dt=0.002):
    _, _, x_final = simulate_projectile(v0, alpha, dt=dt)
    return x_final


def golden_section_max(v0, a_low=1e-6, a_high=math.pi/2 - 1e-6, tol=1e-6, max_iter=100, dt=0.002):
    # Поиск максимума на отрезке [a_low, a_high] методом золотого сечения.
    phi = (math.sqrt(5) - 1) / 2.0
    lo, hi = a_low, a_high

    x1 = hi - phi * (hi - lo)
    x2 = lo + phi * (hi - lo)
    f1 = numeric_range(v0, x1, dt=dt)
    f2 = numeric_range(v0, x2, dt=dt)

    for _ in range(max_iter):
        if abs(hi - lo) < tol:
            break
        if f1 < f2:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + phi * (hi - lo)
            f2 = numeric_range(v0, x2, dt=dt)
        else:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - phi * (hi - lo)
            f1 = numeric_range(v0, x1, dt=dt)

    candidates = [(x1, f1), (x2, f2), ((lo+hi)/2.0, numeric_range(v0, 0.5*(lo+hi), dt=dt))]
    alpha_opt, R_opt = max(candidates, key=lambda p: p[1])
    return float(alpha_opt), float(R_opt)

v0_list = [8.0, 10.0, 12.0]
angles_deg = np.linspace(10, 80, 8)
angles_rad = np.radians(angles_deg)

anim_angles_deg = [15, 30, 45, 60]
anim_angles = np.radians(anim_angles_deg)
v_anim = 10.0
dt_sim = 0.002
    
fig, ax = plt.subplots(figsize=(9, 5))
ax.grid(True)
ax.set_xlabel("x, м")
ax.set_ylabel("y, м")
ax.set_title("Семейство траекторий (разные скорости и углы)")

for v0 in v0_list:
    plotted = False
    for alpha in angles_rad:
        xs, ys, _ = simulate_projectile(v0, alpha, dt=dt_sim)
        label = f"v0={v0:.0f} м/с" if not plotted else None
        ax.plot(xs, ys, lw=1.2, alpha=0.8, label=label)
        plotted = True

v_example = 10.0
alpha_example = math.pi/4
R_an = analytic_range(v_example, alpha_example)
x_an = np.linspace(0, R_an, 300)
y_an = x_an * math.tan(alpha_example) - g * x_an**2 / (2 * (v_example * math.cos(alpha_example))**2)
ax.plot(x_an, y_an, c='black', lw=2, label="Аналитич. (v0=10, α=45°)")

ax.set_ylim(bottom=0)
ax.legend(loc='upper right', fontsize='small')
plt.tight_layout()

fig2, ax2 = plt.subplots(figsize=(9,5))
ax2.grid(True)

trajs = []
for alpha in anim_angles:
    xs, ys, _ = simulate_projectile(v_anim, alpha, dt=dt_sim)
    trajs.append((xs, ys))

max_x = max(xs[-1] for xs, _ in trajs)
max_y = max(np.max(ys) for _, ys in trajs)
ax2.set_xlim(0, 1.05 * max_x)
ax2.set_ylim(0, 1.15 * max_y)
ax2.set_xlabel("x, м")
ax2.set_ylabel("y, м")
ax2.set_title(f"Анимация полёта для углов {anim_angles_deg}° (v0={v_anim} м/с)")

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']
lines = []
dots = []
for i, (xs, ys) in enumerate(trajs):
    line, = ax2.plot([], [], lw=1.8, ls='--', label=f"{anim_angles_deg[i]}°", color=colors[i % len(colors)])
    dot, = ax2.plot([], [], 'o', color=colors[i % len(colors)])
    lines.append(line)
    dots.append(dot)
ax2.legend()

max_len = max(len(xs) for xs, _ in trajs)

def init_anim():
    for line, dot in zip(lines, dots):
        line.set_data([], [])
        dot.set_data([], [])
    return lines + dots

def update_anim(frame):
    for idx, (xs, ys) in enumerate(trajs):
        i = min(frame, len(xs)-1)
        lines[idx].set_data(xs[:i+1], ys[:i+1])
        dots[idx].set_data([xs[i]], [ys[i]])
    return lines + dots

ani = FuncAnimation(fig2, update_anim, frames=max_len, init_func=init_anim,
                    interval=30, blit=True, repeat=False)

v0_opt = 10.0
alpha_num_opt, R_num_opt = golden_section_max(v0_opt, dt=dt_sim, tol=1e-6, max_iter=80)
alpha_analytic = math.pi / 4
R_analytic = analytic_range(v0_opt, alpha_analytic)

print("=== Результаты оптимизации ===")
print(f"Численно найденный оптимальный угол: {math.degrees(alpha_num_opt):.6f}°")
print(f"Аналитический оптимальный угол: {math.degrees(alpha_analytic):.6f}° (== 45°)")
print(f"Численная дальность в этом угле: {R_num_opt:.6f} м")
print(f"Аналитическая дальность при 45°: {R_analytic:.6f} м")
print(f"Абсолютная разница дальностей: {abs(R_num_opt - R_analytic):.6e} м")

xs_a, ys_a, xg_a = simulate_projectile(v0_opt, alpha_analytic, dt=dt_sim)
xs_n, ys_n, xg_n = simulate_projectile(v0_opt, alpha_num_opt, dt=dt_sim)
ax.plot(xs_a, ys_a, c='black', lw=2)
ax.plot(xs_n, ys_n, c='magenta', lw=2, ls='-.', label=f"Числ. opt = {math.degrees(alpha_num_opt):.4f}°")
ax.scatter([xg_a, xg_n], [0, 0], c=['black', 'magenta'], zorder=10)
ax.legend(loc='upper right', fontsize='small')

plt.show()