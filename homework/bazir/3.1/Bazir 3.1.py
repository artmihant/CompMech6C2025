import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, fsolve
from scipy.integrate import solve_ivp
import time

# Параметры задачи
g = 9.81
v0 = 100
target = (500, 50)

def analytical_trajectory(alpha, v0, x0):
    return x0 * np.tan(alpha) - (g * x0**2) / (2 * v0**2 * np.cos(alpha)**2)

def f_newton(alpha):
    return analytical_trajectory(alpha, v0, target[0]) - target[1]

def df_newton(alpha):
    return (target[0] / np.cos(alpha)**2 - 
            (g * target[0]**2 * np.sin(alpha)) / (v0**2 * np.cos(alpha)**3))

def projectile_motion(t, y, alpha):
    x, y_pos, vx, vy = y
    dxdt = vx
    dydt = vy
    dvxdt = 0
    dvydt = -g
    return [dxdt, dydt, dvxdt, dvydt]

def f_shooting(alpha):
    if isinstance(alpha, (list, np.ndarray)):
        alpha = alpha[0]
    y0 = [0, 0, v0 * np.cos(alpha), v0 * np.sin(alpha)]
    t_span = (0, 2 * v0 / g)
    sol = solve_ivp(projectile_motion, t_span, y0, args=(alpha,), 
                    dense_output=True, rtol=1e-8, atol=1e-8)
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    if target[0] > x_sol[-1]:
        return -1000
    y_at_x0 = np.interp(target[0], x_sol, y_sol)
    return y_at_x0 - target[1]

# Решение методом Ньютона
print("Метод Ньютона:")
start_time = time.time()
try:
    sol_newton = root_scalar(f_newton, fprime=df_newton, x0=0.5, method='newton')
    alpha_newton = sol_newton.root
    newton_success = sol_newton.converged
    newton_iterations = sol_newton.iterations
except:
    alpha_newton = None
    newton_success = False
    newton_iterations = 0
time_newton = time.time() - start_time

if alpha_newton is not None:
    print(f"Угол: {alpha_newton:.6f} рад ({np.degrees(alpha_newton):.2f}°)")
    print(f"Итераций: {newton_iterations}")
    print(f"Невязка: {abs(f_newton(alpha_newton)):.2e}")
else:
    print("Метод Ньютона не сошелся")

# Решение методом стрельбы
print("\nМетод стрельбы:")
start_time = time.time()
try:
    sol_shooting = root_scalar(f_shooting, x0=0.5, x1=0.6, method='secant')
    alpha_shooting = sol_shooting.root
    shooting_success = sol_shooting.converged
    shooting_iterations = sol_shooting.iterations
except:
    alpha_shooting = None
    shooting_success = False
    shooting_iterations = 0
time_shooting = time.time() - start_time

if alpha_shooting is not None:
    print(f"Угол: {alpha_shooting:.6f} рад ({np.degrees(alpha_shooting):.2f}°)")
    print(f"Итераций: {shooting_iterations}")
    print(f"Невязка: {abs(f_shooting(alpha_shooting)):.2e}")
else:
    print("Метод стрельбы не сошелся")

# Визуализация траекторий
def plot_trajectory(alpha, label, color):
    t_flight = 2 * v0 * np.sin(alpha) / g
    t = np.linspace(0, t_flight, 100)
    x = v0 * np.cos(alpha) * t
    y = v0 * np.sin(alpha) * t - 0.5 * g * t**2
    plt.plot(x, y, color=color, label=label, linewidth=2)

plt.figure(figsize=(12, 8))

# Рисуем траектории для найденных углов
if alpha_newton is not None:
    plot_trajectory(alpha_newton, 'Метод Ньютона', 'red')

if alpha_shooting is not None:
    plot_trajectory(alpha_shooting, 'Метод стрельбы', 'blue')

#Дополнительные траектории
test_angles = [0.1, 0.3, 0.5, 0.7]
colors = ['gray', 'lightblue', 'lightgreen', 'orange']
for i, angle in enumerate(test_angles):
    plot_trajectory(angle, f'Угол {np.degrees(angle):.1f}°', colors[i])

plt.plot(target[0], target[1], 'ko', markersize=10, label='Цель', markerfacecolor='red')
plt.plot([0], [0], 'go', markersize=8, label='Старт')

plt.xlabel('X (м)')
plt.ylabel('Y (м)')
plt.title('Траектории при различных углах броска')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

print("===================")
print("Сравнение методов")

if alpha_newton is not None and alpha_shooting is not None:
    print(f"Разница в углах: {abs(alpha_newton - alpha_shooting):.2e} рад")
    print(f"Относительная разница: {abs(alpha_newton - alpha_shooting)/alpha_newton*100:.2f}%")
    
    print(f"\nСкорость сходимости:")
    print(f"Метод Ньютона: {newton_iterations} итераций")
    print(f"Метод стрельбы: {shooting_iterations} итераций")
