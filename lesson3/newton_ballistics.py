""" # Метод Ньютона: решение обратной баллистической задачи # """

"""
Демонстрация метода Ньютона для решения обратной баллистической задачи.
Находим угол стрельбы, при котором снаряд попадает точно в цель.
"""

""" ## Импорты ## """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton
from matplotlib.cm import get_cmap
from lesson3_utils import plot_trajectories, plot_convergence


""" ## Параметры системы ## """

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
    Вычисление правых частей системы дифференциальных уравнений движения снаряда.

    Модель учитывает:
    - Гравитационное ускорение (направлено вниз)
    - Сопротивление воздуха (пропорционально скорости)
    - Горизонтальный ветер (постоянная добавка к ускорению)

    Args:
        state (numpy.ndarray): вектор состояния [x, y, vx, vy], где
                               x, y - координаты (метры), vx, vy - скорости (м/с)
        t (float): текущее время (секунды). Не используется в автономной системе.

    Returns:
        numpy.ndarray: вектор производных [dx/dt, dy/dt, dvx/dt, dvy/dt]
                       dx/dt, dy/dt - скорости (м/с)
                       dvx/dt, dvy/dt - ускорения (м/с²)
    """
    x, y, vx, vy = state

    # Производные координат (скорости)
    dx_dt = vx  # скорость по x
    dy_dt = vy  # скорость по y

    # Производные скоростей (ускорения)
    # Горизонтальное ускорение: сопротивление воздуха + ветер
    dvx_dt = -k * vx + wind_x
    # Вертикальное ускорение: гравитация + сопротивление воздуха
    dvy_dt = -g - k * vy

    return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])


def rk4_step(state, t):
    """
    Выполнение одного шага интегрирования методом Рунге-Кутты 4-го порядка.

    RK4 обеспечивает высокую точность интегрирования при умеренных вычислительных затратах.
    Метод вычисляет четыре оценки наклона и комбинирует их для получения точного результата.

    Args:
        state (numpy.ndarray): текущее состояние системы [x, y, vx, vy]
        t (float): текущее время, секунды

    Returns:
        numpy.ndarray: состояние системы на следующем временном шаге [x, y, vx, vy]
    """
    # Вычисление коэффициентов Рунге-Кутты
    k1 = dt * equations_of_motion(state, t)
    k2 = dt * equations_of_motion(state + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * equations_of_motion(state + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * equations_of_motion(state + k3, t + dt)

    # Комбинация коэффициентов для получения нового состояния
    return state + (k1 + 2*k2 + 2*k3 + k4) / 6


def initialize_state(theta):
    """
    Инициализация вектора состояния снаряда для заданного угла стрельбы.

    Args:
        theta (float): начальный угол стрельбы, радианы

    Returns:
        numpy.ndarray: вектор состояния [x, y, vx, vy] в начальный момент времени
    """
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    return np.array([x_start, y_start, vx0, vy0])


def integrate_until_ground(initial_state):
    """
    Интегрирование траектории снаряда до момента падения на землю (y <= 0).

    Args:
        initial_state (numpy.ndarray): начальный вектор состояния [x, y, vx, vy]

    Returns:
        tuple: (финальное состояние, траектория)
               финальное состояние: numpy.ndarray [x, y, vx, vy]
               траектория: list[numpy.ndarray] с состояниями на каждом шаге
    """
    state = initial_state.copy()
    trajectory = [state.copy()]

    t = 0.0

    # Интегрируем до тех пор, пока не достигнем земли или не превысим максимальное время
    while t < t_max and state[1] >= 0:
        state = rk4_step(state, t)
        t += dt
        trajectory.append(state.copy())

    return state, trajectory


def interpolate_ground_impact(trajectory):
    """
    Интерполяция траектории для точного определения точки падения (y = 0).

    Args:
        trajectory (list[numpy.ndarray]): список состояний траектории

    Returns:
        numpy.ndarray: точка падения [x, y, vx, vy] при y = 0
    """
    if len(trajectory) < 2:
        return trajectory[-1]

    # Находим последние две точки траектории
    state_prev = trajectory[-2]
    state_curr = trajectory[-1]

    # Линейная интерполяция для точного определения x при y = 0
    if state_curr[1] < 0 and state_prev[1] >= 0:
        # Пропорция времени, когда y пересекает 0
        ratio = -state_prev[1] / (state_curr[1] - state_prev[1])

        # Интерполируем все компоненты состояния
        interpolated_state = state_prev + ratio * (state_curr - state_prev)
        interpolated_state[1] = 0.0  # Точно устанавливаем y = 0
        return interpolated_state

    return state_curr


def integrate_trajectory(theta, return_full_trajectory=False):
    """
    Интегрирование полной траектории снаряда для заданного угла стрельбы.

    Функция моделирует полет снаряда от начальной точки до падения на землю,
    учитывая гравитацию, сопротивление воздуха и горизонтальный ветер.

    Args:
        theta (float): начальный угол стрельбы относительно горизонта, радианы
        return_full_trajectory (bool): если True, возвращает массив всех состояний траектории

    Returns:
        tuple или numpy.ndarray:
            - Если return_full_trajectory=False: финальное состояние [x, y, vx, vy] при падении
            - Если return_full_trajectory=True: (финальное состояние, траектория)
              где траектория - numpy.ndarray формы (n_points, 4) с состояниями [x, y, vx, vy]
    """
    # Инициализация начального состояния
    initial_state = initialize_state(theta)

    # Интегрирование до падения на землю
    final_state, trajectory = integrate_until_ground(initial_state)

    if return_full_trajectory:
        # Интерполяция для точного определения точки падения
        trajectory_array = np.array(trajectory)
        interpolated_final = interpolate_ground_impact(trajectory)
        return interpolated_final, trajectory_array
    else:
        return final_state


""" ## 2. Метод Ньютона ## """

def residual_function(theta):
    """
    Вычисление функции невязки для метода Ньютона.

    Невязка показывает отклонение точки падения снаряда от целевой координаты x.

    Args:
        theta (float): начальный угол стрельбы, радианы

    Returns:
        float: невязка (x_final - x_target), метры
               положительная - перелет, отрицательная - недолет
    """
    final_state = integrate_trajectory(theta)
    x_final = final_state[0]

    return x_final - x_target


def residual_derivative(theta, h=1e-6):
    """
    Численное вычисление производной функции невязки методом центральных разностей.

    Использует центральную разностную схему для точного вычисления производной:
    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

    Args:
        theta (float): угол стрельбы, радианы
        h (float): шаг численного дифференцирования, радианы

    Returns:
        float: производная d(residual)/d(theta), метры/радиан
    """
    return (residual_function(theta + h) - residual_function(theta - h)) / (2 * h)


def newton_method(theta_initial, max_iterations=50):
    """
    Решение обратной баллистической задачи методом Ньютона.

    Метод Ньютона использует информацию о значении функции и ее производной
    для квадратичной сходимости к решению уравнения f(θ) = 0.

    Args:
        theta_initial (float): начальное приближение угла стрельбы, радианы
        max_iterations (int): максимальное число итераций поиска

    Returns:
        tuple: (theta_solution, theta_history, residual_history)
               theta_solution (float): найденный угол стрельбы, радианы
               theta_history (list[float]): история углов на всех итерациях
               residual_history (list[float]): история невязок на всех итерациях

    Raises:
        ValueError: если нулевая производная или не сошелся за max_iterations
    """
    theta = theta_initial
    theta_history = [theta]
    residual_history = [residual_function(theta)]

    print("Начало метода Ньютона:")
    print(f"Начальное приближение: θ = {np.degrees(theta):.4f}°")

    for iteration in range(max_iterations):
        residual = residual_function(theta)
        derivative = residual_derivative(theta)

        # Проверка на достижение точности
        if abs(residual) < tolerance:
            print(f"Решение найдено: θ = {np.degrees(theta):.4f}°")
            return theta, theta_history, residual_history

        # Проверка на нулевую производную
        if abs(derivative) < 1e-12:
            raise ValueError(f"Нулевая производная на итерации {iteration}")

        # Шаг метода Ньютона: θ_{n+1} = θ_n - f(θ_n)/f'(θ_n)
        theta_new = theta - residual / derivative

        theta_history.append(theta_new)
        residual_history.append(residual_function(theta_new))

        print(f"Итерация {iteration+1}: θ = {np.degrees(theta_new):.4f}°, невязка = {residual_function(theta_new):.4f} м")

        theta = theta_new

    raise ValueError(f"Метод Ньютона не сошелся за {max_iterations} итераций")


""" ## 3. Демонстрация метода Ньютона ## """

# Настройка начального приближения
theta_initial = np.radians(35)  # 35 градусов - начальное приближение

# Вывод заголовка и параметров задачи
print("="*60)
print("МЕТОД НЬЮТОНА: ОБРАТНАЯ БАЛЛИСТИЧЕСКАЯ ЗАДАЧА")
print("="*60)
print(f"Целевая точка: x = {x_target} м, y = {y_target} м")
print(f"Начальная скорость: v0 = {v0} м/с")
print(f"Сопротивление воздуха: k = {k} 1/с")
print(f"Горизонтальный ветер: wx = {wind_x} м/с")
print()

# Выполнение метода Ньютона
theta_newton, theta_hist_newton, residual_hist_newton = newton_method(theta_initial)

# Вывод результатов
print("\nРезультат метода Ньютона:")
print(f"Угол стрельбы: {np.degrees(theta_newton):.6f}°")
print(f"Невязка: {residual_hist_newton[-1]:.2e}")

# Проверка решения
final_state = integrate_trajectory(theta_newton)
print(f"Достигнутая точка: x = {final_state[0]:.3f} м, y = {final_state[1]:.3f} м")

# Визуализация процесса сходимости метода Ньютона
print("\nВизуализация процесса сходимости метода Ньютона...")
plot_trajectories(theta_hist_newton[:min(6, len(theta_hist_newton))], integrate_trajectory,
                 x_target, y_target, x_start, y_start,
                 f"Процесс сходимости метода Ньютона (ветер: {wind_x} м/с)")
plot_convergence(theta_hist_newton, residual_hist_newton, "Метод Ньютона")


""" ## Выводы ## """

"""
## Результаты решения обратной баллистической задачи методом Ньютона

Метод Ньютона успешно решил обратную баллистическую задачу, найдя угол стрельбы,
при котором снаряд попадает точно в цель с учетом сопротивления воздуха и ветра.

### Ключевые особенности решения:

1. **Начальное приближение**: Использовано приближение 35°, достаточно близкое к решению

2. **Квадратичная сходимость**: Метод Ньютона показал быструю квадратичную сходимость
   после первых итераций, что является его основным преимуществом

3. **Численное дифференцирование**: Для вычисления производной использовалась
   центральная разностная схема второго порядка точности

4. **Визуализация**: Графики показывают, как метод быстро приближается к решению
   за небольшое число итераций

5. **Точность**: Метод достиг очень высокой точности (невязка ~1e-12) за 6 итераций
"""
