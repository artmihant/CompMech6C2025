""" # Обратная баллистическая задача: метод стрельбы # """

"""
На этом занятии мы рассмотрим практическое применение метода стрельбы для решения
обратной баллистической задачи. Представьте, что мы хотим попасть снарядом точно
в заданную точку на поверхности земли, зная начальную скорость выстрела, но не зная
точный угол прицеливания.

## Физическая постановка задачи

Тело движется в двумерном пространстве под действием:
- Гравитационной силы (ускорение свободного падения g)
- Силы сопротивления воздуха (пропорциональной скорости)
- Постоянного горизонтального ветра

Уравнения движения в проекциях:
- dx/dt = vx
- dy/dt = vy
- dvx/dt = -k*vx + wx  (сопротивление + ветер)
- dvy/dt = -g - k*vy

Начальные условия: x(0) = 0, y(0) = 0, vx(0) = v0*cos(θ), vy(0) = v0*sin(θ)
Конечные условия: x(T) = xtarget, y(T) = 0

Нужно найти угол θ такой, чтобы траектория заканчивалась в точке (xtarget, 0).
"""

""" ## Импорты и параметры системы ## """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton
from matplotlib.cm import get_cmap

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


""" ## 1. Определяем систему уравнений движения ## """

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
    dvx_dt = -k * vx + wind_x
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

# Выполнение и проверка базовой функциональности
print("Проверка базовой функциональности системы уравнений:")
test_theta = np.radians(45)
final_state = integrate_trajectory(test_theta)
print(f"Тестовый угол 45°: достигает точки x = {final_state[0]:.1f} м")


""" ## 2. Решаем задачу методом стрельбы ## """

def residual_function(theta):
    """
    Функция невязки для метода стрельбы

    Args:
        theta: начальный угол стрельбы, радианы

    Returns:
        float: отклонение конечной координаты x от целевой
    """
    final_state = integrate_trajectory(theta)
    x_final = final_state[0]

    return x_final - x_target


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
    residual_left = residual_function(theta_left)
    residual_right = residual_function(theta_right)

    theta_history = [theta_left, theta_right]
    residual_history = [residual_left, residual_right]

    print("Начало метода стрельбы:")
    print(f"θ_left = {np.degrees(theta_left):.4f}, residual_left = {residual_left:.4f}")
    print(f"θ_right = {np.degrees(theta_right):.4f}, residual_right = {residual_right:.4f}")

    for iteration in range(max_iterations):
        # Проверка на достижение точности
        if abs(residual_left) < tolerance:
            print(f"Решение найдено: θ = {np.degrees(theta_left):.4f}")
            return theta_left, theta_history, residual_history

        if abs(residual_right) < tolerance:
            print(f"Решение найдено: θ = {np.degrees(theta_right):.4f}")
            return theta_right, theta_history, residual_history

        # Проверка на сходимость
        if abs(residual_right - residual_left) < tolerance:
            raise ValueError("Метод не сошелся: невязки слишком близки")

        # Вычисление нового угла по методу regula falsi
        theta_new = (residual_right * theta_left - residual_left * theta_right) / (residual_right - residual_left)

        residual_new = residual_function(theta_new)

        theta_history.append(theta_new)
        residual_history.append(residual_new)

        print(f"Итерация {iteration+1}: θ = {np.degrees(theta_new):.4f}, невязка = {residual_new:.4f}")

        # Выбор новой пары точек
        if residual_right * residual_left > 0:
            # Невязки одного знака - выбираем точку ближе к новой
            if abs(theta_new - theta_left) > abs(theta_new - theta_right):
                theta_left, residual_left = theta_new, residual_new
            else:
                theta_right, residual_right = theta_new, residual_new
        else:
            # Невязки разных знаков - выбираем точку с противоположным знаком
            if residual_new * residual_left > 0:
                theta_left, residual_left = theta_new, residual_new
            else:
                theta_right, residual_right = theta_new, residual_new

    raise ValueError(f"Метод не сошелся за {max_iterations} итераций")

# Выполнение метода стрельбы
theta_left = np.radians(10)   # 10 градусов - снаряд не долетит
theta_right = np.radians(80)  # 80 градусов - снаряд перелетит

try:
    theta_shooting, theta_hist_shooting, residual_hist_shooting = shooting_method(theta_left, theta_right)

    print("\nРезультат метода стрельбы:")
    print(f"Угол стрельбы: {np.degrees(theta_shooting):.6f}°")
    print(f"Невязка: {residual_hist_shooting[-1]:.2e}")

    # Проверка решения
    final_state = integrate_trajectory(theta_shooting)
    print(f"Достигнутая точка: x = {final_state[0]:.3f} м, y = {final_state[1]:.3f} м")

except ValueError as e:
    print(f"Ошибка в методе стрельбы: {e}")
    theta_shooting = None


""" ## 4. Решаем задачу методом Ньютона ## """

def residual_derivative(theta, h=1e-6):
    """
    Численное вычисление производной функции невязки

    Args:
        theta: угол стрельбы, радианы
        h: шаг для численного дифференцирования

    Returns:
        float: производная d(residual)/d(theta)
    """
    return (residual_function(theta + h) - residual_function(theta - h)) / (2 * h)


def newton_method(theta_initial, max_iterations=50):
    """
    Метод Ньютона для решения баллистической задачи

    Args:
        theta_initial: начальное приближение угла, радианы
        max_iterations: максимальное число итераций

    Returns:
        tuple: (найденный угол, список углов на итерациях, список невязок)
    """
    theta = theta_initial
    theta_history = [theta]
    residual_history = [residual_function(theta)]

    print("\nНачало метода Ньютона:")
    print(f"Начальное приближение: θ = {np.degrees(theta):.4f}")

    for iteration in range(max_iterations):
        residual = residual_function(theta)
        derivative = residual_derivative(theta)

        # Проверка на достижение точности
        if abs(residual) < tolerance:
            print(f"Решение найдено: θ = {np.degrees(theta):.4f}")
            return theta, theta_history, residual_history

        # Проверка на нулевую производную
        if abs(derivative) < 1e-12:
            raise ValueError(f"Нулевая производная на итерации {iteration}")

        # Шаг метода Ньютона
        theta_new = theta - residual / derivative

        theta_history.append(theta_new)
        residual_history.append(residual_function(theta_new))

        print(f"Итерация {iteration+1}: θ = {np.degrees(theta_new):.4f}, невязка = {residual_function(theta_new):.4f}")

        theta = theta_new

    raise ValueError(f"Метод Ньютона не сошелся за {max_iterations} итераций")

# Выполнение метода Ньютона
theta_initial = np.radians(35)  # 35 градусов - начальное приближение

try:
    theta_newton, theta_hist_newton, residual_hist_newton = newton_method(theta_initial)

    print("\nРезультат метода Ньютона:")
    print(f"Угол стрельбы: {np.degrees(theta_newton):.6f}°")
    print(f"Невязка: {residual_hist_newton[-1]:.2e}")

    # Проверка решения
    final_state = integrate_trajectory(theta_newton)
    print(f"Достигнутая точка: x = {final_state[0]:.3f} м, y = {final_state[1]:.3f} м")

except ValueError as e:
    print(f"Ошибка в методе Ньютона: {e}")
    theta_newton = None


""" ## 5. Создаем функции визуализации ## """

def plot_trajectories(theta_list, title="Траектории для разных углов стрельбы", custom_labels=None):
    """
    Визуализация траекторий для списка углов

    Args:
        theta_list: список углов в радианах
        title: заголовок графика
        custom_labels: пользовательские метки для легенд (если None, используются углы)
    """
    plt.figure(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(theta_list)))

    for i, theta in enumerate(theta_list):
        final_state, trajectory = integrate_trajectory(theta, return_full_trajectory=True)

        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]

        # Определяем метку для легенды
        if custom_labels is not None and i < len(custom_labels):
            label = custom_labels[i]
        else:
            label = f'{np.degrees(theta):.1f}°'

        plt.plot(x_coords, y_coords, color=colors[i], label=label)

    # Целевая точка
    plt.scatter([x_target], [y_target], color='red', s=100, marker='x',
               label=f'Цель ({x_target:.1f}, {y_target:.1f})', zorder=5)

    # Стартовая точка
    plt.scatter([x_start], [y_start], color='green', s=100, marker='o',
               label=f'Старт ({x_start:.1f}, {y_start:.1f})', zorder=5)

    plt.xlabel('Расстояние, м')
    plt.ylabel('Высота, м')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_convergence(theta_history, residual_history, method_name):
    """
    Визуализация процесса сходимости метода

    Args:
        theta_history: история изменения угла
        residual_history: история изменения невязки
        method_name: название метода для заголовка
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    iterations = range(len(theta_history))

    # График изменения угла
    ax1.plot(iterations, np.degrees(theta_history), 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Угол, градусы')
    ax1.set_title(f'Сходимость угла ({method_name})')
    ax1.grid(True, alpha=0.3)

    # График изменения невязки
    ax2.plot(iterations, residual_history, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Итерация')
    ax2.set_ylabel('Невязка, м')
    ax2.set_title(f'Сходимость невязки ({method_name})')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


""" ## 6. Визуализация первых траекторий метода стрельбы ## """

# Показываем первые несколько траекторий из истории метода стрельбы
if theta_shooting is not None:
    n_trajectories = min(8, len(theta_hist_shooting))  # Показываем до 8 траекторий
    plot_trajectories(theta_hist_shooting[:n_trajectories],
                     f"Первые {n_trajectories} траекторий метода стрельбы (ветер: {wind_x} м/с)")
else:
    # Если метод не сошелся, показываем хотя бы начальные траектории
    initial_thetas = [theta_left, theta_right]
    plot_trajectories(initial_thetas, f"Начальные траектории метода стрельбы (ветер: {wind_x} м/с)")

# Выполнение визуализации результатов
print("\nВизуализация результатов...")

# Диапазон углов для демонстрации
theta_range = np.radians(np.linspace(10, 60, 11))  # 10° to 60°

# Траектории для диапазона углов
plot_trajectories(theta_range, f"Траектории для разных углов стрельбы (ветер: {wind_x} м/с)")


""" ## 7. Визуализируем процесс сходимости методов ## """

# Траектории процесса сходимости метода стрельбы
if theta_shooting is not None:
    plot_trajectories(theta_hist_shooting[:min(6, len(theta_hist_shooting))],
                     f"Процесс сходимости метода стрельбы (ветер: {wind_x} м/с)")
    plot_convergence(theta_hist_shooting, residual_hist_shooting, "Метод стрельбы")

# Траектории процесса сходимости метода Ньютона
if theta_newton is not None:
    plot_trajectories(theta_hist_newton[:min(6, len(theta_hist_newton))],
                     f"Процесс сходимости метода Ньютона (ветер: {wind_x} м/с)")
    plot_convergence(theta_hist_newton, residual_hist_newton, "Метод Ньютона")


""" ## 8. Сравниваем найденные решения ## """

# Финальные траектории
final_thetas = []
labels = []
if theta_shooting is not None:
    final_thetas.append(theta_shooting)
    labels.append("Метод стрельбы")
if theta_newton is not None:
    final_thetas.append(theta_newton)
    labels.append("Метод Ньютона")

if final_thetas:
    plot_trajectories(final_thetas, f"Сравнение найденных решений (ветер: {wind_x} м/с)", custom_labels=labels)


""" ## Выводы ## """

"""
## Результаты решения обратной баллистической задачи

В этом уроке мы рассмотрели практическое применение метода стрельбы для решения
обратной баллистической задачи - определения начального угла стрельбы, при котором
снаряд попадает точно в заданную точку.

### Ключевые аспекты решения:

1. **Физическая модель**: Учли гравитацию, сопротивление воздуха и постоянный ветер

2. **Численный метод**: Использовали метод Рунге-Кутты 4-го порядка для интегрирования траекторий

3. **Метод стрельбы**: Применили regula falsi для решения краевой задачи

4. **Альтернативный метод**: Реализовали метод Ньютона как более прямолинейный подход

5. **Визуализация**: Показали процесс сходимости и сравнили различные траектории

### Практические выводы:

- Метод стрельбы гарантированно сходится при правильном выборе начальных границ
- Метод Ньютона может быть быстрее, но требует хорошего начального приближения
- Ветер и сопротивление воздуха существенно влияют на траекторию
- Визуализация помогает понять физический смысл решения

Эта задача демонстрирует, как абстрактные математические методы находят применение
в реальных инженерных расчетах, таких как артиллерийская стрельба или ракетная техника.
"""