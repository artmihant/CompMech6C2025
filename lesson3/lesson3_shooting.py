""" # Метод стрельбы: решение обратной баллистической задачи # """

"""
Демонстрация метода стрельбы для решения обратной баллистической задачи.
Находим угол стрельбы, при котором снаряд попадает точно в цель.
"""

""" ## Импорты и параметры ## """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton
from matplotlib.cm import get_cmap
from lesson3_utils import plot_trajectories, plot_convergence

# Ускорение свободного падения, м/с²
g = 9.81

# Коэффициент сопротивления воздуха, 1/с
k = 0.01

# Скорость горизонтального ветра, м/с
wind_x = 2.0

# Начальная скорость снаряда, м/с
v0 = 50.0

# Целевая точка (x_target, y_target), м
x_target = 200.0
y_target = 0.0

# Начальная точка (x_start, y_start), м
x_start = 0.0
y_start = 0.0

# Точность решения
tolerance = 1e-6

# Максимальное время полета для поиска, с
t_max = 20.0

# Число шагов интегрирования
n_steps = 1000

# Шаг по времени
dt = t_max / n_steps


""" ## 1. Система уравнений движения ## """

def equations_of_motion(state, t):
    """
    Правые части системы уравнений движения снаряда

    Args:
        state: вектор состояния [x, y, vx, vy]
        t: время (не используется в автономной системе)

    Returns:
        array: производные [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    x, y, vx, vy = state

    # Производные координат
    dx_dt = vx
    dy_dt = vy

    # Производные скоростей (сопротивление воздуха + ветер + гравитация)
    dvx_dt = -k * (vx - wind_x)
    dvy_dt = -g - k * vy

    return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])


def rk4_step(state, t):
    """
    Один шаг метода Рунге-Кутты 4-го порядка

    Args:
        state: текущее состояние системы
        t: текущее время

    Returns:
        array: состояние системы на следующем шаге
    """
    k1 = dt * equations_of_motion(state, t)
    k2 = dt * equations_of_motion(state + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * equations_of_motion(state + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * equations_of_motion(state + k3, t + dt)

    return state + (k1 + 2*k2 + 2*k3 + k4) / 6


def integrate_trajectory(theta, return_full_trajectory=False):
    """
    Интегрирование траектории снаряда для заданного угла theta

    Args:
        theta: начальный угол стрельбы, радианы
        return_full_trajectory: если True, возвращает всю траекторию

    Returns:
        tuple: (конечная точка, полная траектория если запрошена)
    """
    # Начальные условия
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    state = np.array([x_start, y_start, vx0, vy0])

    if return_full_trajectory:
        trajectory = [state.copy()]

    t = 0.0

    # Интегрируем до тех пор, пока не достигнем земли (y <= 0) или не превысим время
    while t < t_max and state[1] >= 0:
        state = rk4_step(state, t)
        t += dt

        if return_full_trajectory:
            trajectory.append(state.copy())

    # Интерполяция для точного попадания в y = 0
    if return_full_trajectory:
        trajectory = np.array(trajectory)
        return state, trajectory
    else:
        return state


""" ## 2. Метод стрельбы ## """

def residual_function(theta):
    """
    Функция невязки для метода стрельбы

    Args:
        theta: начальный угол стрельбы, радианы

    Returns:
        float: отклонение конечной координаты x от целевой
    """
    # TODO 

    return 0


def shooting_method(theta_left, theta_right, max_iterations=50):
    """
    Метод стрельбы (regula falsi) для решения баллистической задачи

    Args:
        theta_left: левая граница поиска угла, радианы
        theta_right: правая граница поиска угла, радианы
        max_iterations: максимальное число итераций

    Returns:
        tuple: (найденный угол, список углов на итерациях, список невязок)
    """
  
    # TODO 
    return None


""" ## 3. Демонстрация метода стрельбы ## """

# Выполнение метода стрельбы
theta_left = np.radians(10)   # 10 градусов - снаряд не долетит
theta_right = np.radians(80)  # 80 градусов - снаряд перелетит

print("="*60)
print("МЕТОД СТРЕЛЬБЫ: ОБРАТНАЯ БАЛЛИСТИЧЕСКАЯ ЗАДАЧА")
print("="*60)
print(f"Целевая точка: x = {x_target} м")
print(f"Начальная скорость: v0 = {v0} м/с")
print(f"Сопротивление воздуха: k = {k} 1/с")
print(f"Горизонтальный ветер: wx = {wind_x} м/с")
print()

try:
    theta_shooting, theta_hist_shooting, residual_hist_shooting = shooting_method(theta_left, theta_right)

    print("\nРезультат метода стрельбы:")
    print(f"Угол стрельбы: {np.degrees(theta_shooting):.6f}°")
    print(f"Невязка: {residual_hist_shooting[-1]:.2e}")

    # Проверка решения
    final_state = integrate_trajectory(theta_shooting)
    print(f"Достигнутая точка: x = {final_state[0]:.3f} м, y = {final_state[1]:.3f} м")

    # Визуализация первых траекторий
    print("\nВизуализация первых траекторий метода стрельбы...")
    n_trajectories = min(8, len(theta_hist_shooting))
    plot_trajectories(theta_hist_shooting[:n_trajectories], integrate_trajectory,
                     x_target, y_target, x_start, y_start,
                     f"Первые {n_trajectories} траекторий метода стрельбы (ветер: {wind_x} м/с)")

    # Визуализация процесса сходимости
    plot_convergence(theta_hist_shooting, residual_hist_shooting, "Метод стрельбы")

except ValueError as e:
    print(f"Ошибка в методе стрельбы: {e}")
    # Если метод не сошелся, показываем хотя бы начальные траектории
    initial_thetas = [theta_left, theta_right]
    plot_trajectories(initial_thetas, integrate_trajectory, x_target, y_target, x_start, y_start,
                     f"Начальные траектории метода стрельбы (ветер: {wind_x} м/с)")

