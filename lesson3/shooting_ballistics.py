""" # Метод стрельбы: решение обратной баллистической задачи # """

"""
Демонстрация двух вариантов метода стрельбы для решения обратной баллистической задачи:
1. Метод стрельбы с корректировкой regula falsi
2. Метод стрельбы с корректировкой методом Ньютона

Находим угол стрельбы, при котором снаряд попадает точно в цель.
"""

""" ## Импорты ## """

import numpy as np
from lesson3_utils import plot_combined_analysis, plot_convergence


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
    - Горизонтальный ветер

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
    dvx_dt = -k * (vx - wind_x)
    # Вертикальное ускорение: гравитация + сопротивление воздуха
    dvy_dt = -g - k * vy

    return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])


def rk4_step(state, t):
    """
    Один шаг метода Рунге-Кутты 4-го порядка

    Args:
        state (numpy.ndarray): текущее состояние системы [x, y, vx, vy]
        t (float): текущее время, секунды

    Returns:
        numpy.ndarray: состояние системы на следующем временном шаге [x, y, vx, vy]
    """
    k1 = dt * equations_of_motion(state, t)
    k2 = dt * equations_of_motion(state + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * equations_of_motion(state + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * equations_of_motion(state + k3, t + dt)

    return state + (k1 + 2*k2 + 2*k3 + k4) / 6


""" ## 2. Функции интегрирования траектории ## """

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
        return_trajectory (bool): если True, возвращает всю траекторию

    Returns:
        tuple: траектория: numpy.ndarray[[x, y, vx, vy]]
    """
    state = initial_state.copy()
    trajectory = [state.copy()]

    t = 0.0

    # Интегрируем до тех пор, пока не достигнем земли или не превысим максимальное время
    while t < t_max and state[1] >= 0:
        state = rk4_step(state, t)
        t += dt

        trajectory.append(state.copy())

    return np.array(trajectory)


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
        return_full_trajectory (bool): если True, возвращает (финальное состояние, траектория)

    Returns:
        Если return_full_trajectory=False: trajectory numpy.ndarray формы (n_points, 4)
        Если return_full_trajectory=True: (финальное состояние, траектория)
    """
    # Инициализация начального состояния
    initial_state = initialize_state(theta)

    # Интегрирование до падения на землю
    trajectory = integrate_until_ground(initial_state)

    # Интерполяция для точного определения точки падения
    interpolated_final = interpolate_ground_impact(trajectory)
    trajectory[-1] = interpolated_final

    if return_full_trajectory:
        return interpolated_final, trajectory
    else:
        return trajectory


""" ## 3. Метод стрельбы с корректировкой regula falsi ## """

def residual_function(theta):
    """
    Вычисление функции невязки для метода стрельбы.

    Невязка показывает отклонение точки падения снаряда от целевой координаты x.

    Args:
        theta (float): начальный угол стрельбы, радианы

    Returns:
        float: невязка (x_final - x_target), метры
               положительная - перелет, отрицательная - недолет
    """
    final_state, trajectory = integrate_trajectory(theta, return_full_trajectory=True)
    x_final = final_state[0]

    return x_final - x_target


def initialize_shooting_bounds(theta_left, theta_right):
    """
    Инициализация границ поиска для метода стрельбы.

    Args:
        theta_left (float): левая граница угла стрельбы, радианы
        theta_right (float): правая граница угла стрельбы, радианы

    Returns:
        tuple: (residual_left, residual_right, theta_history, residual_history)
               где theta_history и residual_history содержат начальные значения
    """
    residual_left = residual_function(theta_left)
    residual_right = residual_function(theta_right)

    theta_history = [theta_left, theta_right]
    residual_history = [residual_left, residual_right]

    return residual_left, residual_right, theta_history, residual_history


def check_convergence(residual_left, residual_right, tolerance):
    """
    Проверка условий сходимости метода стрельбы.

    Args:
        residual_left (float): невязка левой границы
        residual_right (float): невязка правой границы
        tolerance (float): требуемая точность решения

    Returns:
        tuple: (converged, theta_solution)
               converged - True если решение найдено
               theta_solution - найденный угол или None
    """
    if abs(residual_left) < tolerance:
        return True, None  # Решение в левой границе
    if abs(residual_right) < tolerance:
        return True, None  # Решение в правой границе
    if abs(residual_right - residual_left) < tolerance:
        raise ValueError("Метод не сошелся: невязки слишком близки")

    return False, None


def compute_new_theta_regula_falsi(theta_left, theta_right, residual_left, residual_right):
    """
    Вычисление нового угла методом regula falsi (ложной позиции).

    Args:
        theta_left (float): левая граница угла, радианы
        theta_right (float): правая граница угла, радианы
        residual_left (float): невязка левой границы
        residual_right (float): невязка правой границы

    Returns:
        float: новый угол стрельбы, радианы
    """
    return (residual_right * theta_left - residual_left * theta_right) / (residual_right - residual_left)


def update_bounds_regula_falsi(theta_left, theta_right, residual_left, residual_right,
                               theta_new, residual_new):
    """
    Обновление границ поиска методом regula falsi.

    Выбирает новую пару точек в зависимости от знаков невязок для обеспечения сходимости.

    Args:
        theta_left (float): текущая левая граница, радианы
        theta_right (float): текущая правая граница, радианы
        residual_left (float): невязка левой границы
        residual_right (float): невязка правой границы
        theta_new (float): новый кандидат угла, радианы
        residual_new (float): невязка нового кандидата

    Returns:
        tuple: (новая_левая_граница, новая_правая_граница,
                новая_невязка_левой, новая_невязка_правой)
    """
    if residual_right * residual_left > 0:
        # Невязки одного знака - выбираем точку ближе к новой
        if abs(theta_new - theta_left) > abs(theta_new - theta_right):
            return theta_new, theta_right, residual_new, residual_right
        else:
            return theta_left, theta_new, residual_left, residual_new
    else:
        # Невязки разных знаков - выбираем точку с противоположным знаком
        if residual_new * residual_left > 0:
            return theta_new, theta_right, residual_new, residual_right
        else:
            return theta_left, theta_new, residual_left, residual_new


def shooting_method(theta_left, theta_right, max_iterations=50):
    """
    Решение обратной баллистической задачи методом стрельбы с использованием regula falsi.

    Метод итеративно уточняет угол стрельбы, чтобы снаряд попадал точно в цель,
    используя комбинацию методов бисекции и секущих для быстрой сходимости.

    Args:
        theta_left (float): левая граница поиска угла стрельбы, радианы
        theta_right (float): правая граница поиска угла стрельбы, радианы
        max_iterations (int): максимальное число итераций поиска

    Returns:
        tuple: (theta_solution, theta_history, residual_history)
               theta_solution (float): найденный угол стрельбы, радианы
               theta_history (list[float]): история углов на всех итерациях
               residual_history (list[float]): история невязок на всех итерациях

    Raises:
        ValueError: если метод не сошелся за заданное число итераций
    """
    # Инициализация границ и истории
    residual_left, residual_right, theta_history, residual_history = initialize_shooting_bounds(
        theta_left, theta_right)

    print("Начало метода стрельбы:")
    print(f"θ_left = {np.degrees(theta_left):.4f}°, residual_left = {float(residual_left):.4f} м")
    print(f"θ_right = {np.degrees(theta_right):.4f}°, residual_right = {float(residual_right):.4f} м")

    # Основной цикл итераций
    for iteration in range(max_iterations):
        # Проверка условий сходимости
        converged, theta_solution = check_convergence(residual_left, residual_right, tolerance)

        if converged:
            if theta_solution is None:  # Решение найдено в одной из границ
                theta_solution = theta_left if abs(residual_left) < tolerance else theta_right
            print(f"Решение найдено: θ = {np.degrees(theta_solution):.4f}°")
            return theta_solution, theta_history, residual_history

        # Вычисление нового кандидата угла
        theta_new = compute_new_theta_regula_falsi(theta_left, theta_right, residual_left, residual_right)
        residual_new = residual_function(theta_new)

        # Сохранение в истории
        theta_history.append(theta_new)
        residual_history.append(residual_new)

        print(f"Итерация {iteration+1}: θ = {np.degrees(theta_new):.4f}°, невязка = {residual_new:.4f} м")

        # Обновление границ поиска
        theta_left, theta_right, residual_left, residual_right = update_bounds_regula_falsi(
            theta_left, theta_right, residual_left, residual_right, theta_new, residual_new)

    raise ValueError(f"Метод не сошелся за {max_iterations} итераций")


""" ## 4. Метод стрельбы с корректировкой методом Ньютона ## """

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


def shooting_method_newton_correction(theta_initial, max_iterations=50):
    """
    Решение обратной баллистической задачи методом стрельбы с корректировкой методом Ньютона.

    Метод стрельбы использует начальное приближение угла и корректирует его
    методом Ньютона, используя информацию о значении функции и ее производной
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

    print("Начало метода стрельбы с корректировкой Ньютона:")
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

    raise ValueError(f"Метод стрельбы с корректировкой Ньютона не сошелся за {max_iterations} итераций")


""" ## 5. Демонстрация методов стрельбы ## """

# Настройка границ поиска для метода стрельбы с regula falsi
theta_left = np.radians(10)   # 10 градусов - снаряд не долетит
theta_right = np.radians(45)  # 45 градусов - снаряд перелетит

# Настройка начального приближения для метода стрельбы с Ньютона
theta_initial = np.radians(35)  # 35 градусов - начальное приближение

# Вывод заголовка и параметров задачи
print("="*60)
print("МЕТОДЫ СТРЕЛЬБЫ: ОБРАТНАЯ БАЛЛИСТИЧЕСКАЯ ЗАДАЧА")
print("="*60)
print(f"Целевая точка: x = {x_target} м, y = {y_target} м")
print(f"Начальная скорость: v0 = {v0} м/с")
print(f"Сопротивление воздуха: k = {k} 1/с")
print(f"Горизонтальный ветер: wx = {wind_x} м/с")
print()

# Выполнение метода стрельбы с regula falsi
print("="*50)
print("МЕТОД СТРЕЛЬБЫ С КОРРЕКТИРОВКОЙ REGULA FALSI")
print("="*50)
theta_shooting, theta_hist_shooting, residual_hist_shooting = shooting_method(theta_left, theta_right)

# Вывод результатов метода стрельбы с regula falsi
print("\nРезультат метода стрельбы с regula falsi:")
print(f"Угол стрельбы: {np.degrees(theta_shooting):.6f}°")
print(f"Невязка: {residual_hist_shooting[-1]:.2e}")

# Проверка решения метода стрельбы с regula falsi
final_state = integrate_trajectory(theta_shooting)[-1]
print(f"Достигнутая точка: x = {final_state[0]:.3f} м, y = {final_state[1]:.3f} м")

# Объединенная визуализация метода стрельбы с regula falsi
print("\nОбъединенная визуализация метода стрельбы с regula falsi...")
n_trajectories = min(8, len(theta_hist_shooting))
plot_combined_analysis(theta_hist_shooting[:n_trajectories], integrate_trajectory,
                      theta_hist_shooting, residual_hist_shooting,
                      x_target, y_target, x_start, y_start,
                      "Метод стрельбы с regula falsi")

print("\n" + "="*50)
print("МЕТОД СТРЕЛЬБЫ С КОРРЕКТИРОВКОЙ МЕТОДОМ НЬЮТОНА")
print("="*50)

# Выполнение метода стрельбы с корректировкой Ньютона
theta_newton, theta_hist_newton, residual_hist_newton = shooting_method_newton_correction(theta_initial)

# Вывод результатов метода стрельбы с Ньютона
print("\nРезультат метода стрельбы с корректировкой Ньютона:")
print(f"Угол стрельбы: {np.degrees(theta_newton):.6f}°")
print(f"Невязка: {residual_hist_newton[-1]:.2e}")

# Проверка решения метода стрельбы с Ньютона
final_state = integrate_trajectory(theta_newton)[-1]
print(f"Достигнутая точка: x = {final_state[0]:.3f} м, y = {final_state[1]:.3f} м")

# Объединенная визуализация метода стрельбы с корректировкой Ньютона
print("\nОбъединенная визуализация метода стрельбы с корректировкой Ньютона...")
plot_combined_analysis(theta_hist_newton[:min(6, len(theta_hist_newton))], integrate_trajectory,
                      theta_hist_newton, residual_hist_newton,
                      x_target, y_target, x_start, y_start,
                      "Метод стрельбы с корректировкой Ньютона")

# Сравнение результатов
print("\n" + "="*50)
print("СРАВНЕНИЕ МЕТОДОВ СТРЕЛЬБЫ")
print("="*50)
print(f"Угол стрельбы (regula falsi): {np.degrees(theta_shooting):.6f}°")
print(f"Угол стрельбы (Ньютон): {np.degrees(theta_newton):.6f}°")
print(f"Разница углов: {abs(np.degrees(theta_shooting - theta_newton)):.2e}°")
print(f"Итераций (regula falsi): {len(theta_hist_shooting)}")
print(f"Итераций (Ньютон): {len(theta_hist_newton)}")


""" ## Выводы ## """

"""
## Результаты решения обратной баллистической задачи методами стрельбы

Оба варианта метода стрельбы успешно решили обратную баллистическую задачу,
найдя угол стрельбы, при котором снаряд попадает точно в цель с учетом
сопротивления воздуха и ветра.

### Сравнение методов стрельбы:

#### Метод стрельбы с корректировкой regula falsi:
- **Принцип работы**: Итеративное уточнение угла путем "пристрелки" с разных позиций
  в заданном интервале, используя метод regula falsi для вычисления следующего кандидата
- **Достоинства**:
  - Не требует начального приближения - нужны только границы интервала
  - Гарантированная сходимость при правильном выборе границ (с разными знаками невязки)
  - Устойчив к локальным экстремумам
- **Недостатки**:
  - Требует большего числа итераций по сравнению с методом Ньютона
  - Медленнее сходится на последних итерациях

#### Метод стрельбы с корректировкой методом Ньютона:
- **Принцип работы**: Использует начальное приближение угла и корректирует его
  методом Ньютона, используя информацию о значении функции невязки и ее производной
  для квадратичной сходимости к решению уравнения f(θ) = 0
- **Достоинства**:
  - Быстрая квадратичная сходимость после первых итераций
  - Минимальное число итераций для достижения высокой точности
- **Недостатки**:
  - Требует хорошего начального приближения
  - Численное дифференцирование может быть неустойчивым
  - Может расходиться при неудачном выборе начального приближения

### Отличие от "классического" метода Ньютона для краевых задач:

Классический метод Ньютона для решения краевых задач механики обычно подразумевает
многошаговый алгоритм, включающий:
- Дискретизацию задачи (метод конечных разностей или конечных элементов)
- Решение системы нелинейных уравнений большой размерности
- Итеративное уточнение решения с использованием якобиана системы

В данном случае мы имеем простую реализацию метода стрельбы с корректировкой Ньютона,
где метод Ньютона применяется к одномерной функции невязки, а не к полной системе
дифференциальных уравнений.

### Практические рекомендации:

1. **Выбор метода**:
   - **Метод стрельбы с regula falsi**: Когда нет хорошего начального приближения
     или нужно гарантированное решение (требуются границы с разными знаками невязки)
   - **Метод стрельбы с Ньютона**: Когда есть хорошее начальное приближение
     и важна быстрая сходимость

2. **Оптимизация**:
   - Оба метода достигают высокой точности (невязка ~1e-10)
   - Метод с корректировкой Ньютона обычно требует меньше вычислений при хорошем старте
   - Метод с regula falsi более надежен при отсутствии априорной информации

3. **Визуализация**:
   - Графики траекторий показывают эволюцию решения
   - Графики сходимости демонстрируют скорость приближения к точному решению
   - Сравнение методов позволяет выбрать оптимальный подход для конкретной задачи
"""
