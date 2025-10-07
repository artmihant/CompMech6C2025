import numpy as np
import matplotlib.pyplot as plt

# Параметры стержня
E = 2.0e11  # модуль Юнга (Па)
J = 5.0e-8  # момент инерции сечения (м^4)
L = 2.0  # длина стержня (м)
q = 1000.0  # равномерно распределенная нагрузка (Н/м)

# Жесткость стержня с линейной зависимостью от растяжения
EJ_0 = E * J  # начальная жесткость
alpha = 0.1  # коэффициент зависимости жесткости от растяжения

# Граничные условия
w_0 = 0.0  # прогиб на левом конце (м)
w_L = 0.0  # прогиб на правом конце (м)

# Параметры интегрирования
x_start = 0.0  # начальная координата (м)
x_end = L  # конечная координата (м)
N = 100  # количество точек разбиения
dx = (x_end - x_start) / N  # шаг по координате (м)

# Создаем массив координат
x = np.linspace(x_start, x_end, N + 1)


def calculate_moment(x_coord, force, length):
    """
    Вычисление изгибающего момента в сечении стержня

    Args:
        x_coord (float): координата сечения (м)
        force (float): интенсивность распределенной нагрузки (Н/м)
        length (float): длина стержня (м)

    Returns:
        float: изгибающий момент (Н·м)
    """
    return force * (length * x_coord - x_coord ** 2) * 0.5


def system_equations(state, x_coord):
    """
    Система дифференциальных уравнений для изгиба стержня

    Уравнения:
    dw/dx = θ (угол поворота)
    dθ/dx = M(x)/(EJ) (кривизна)

    Args:
        state (numpy.ndarray): вектор состояния [w, θ]
        x_coord (float): координата сечения (м)

    Returns:
        numpy.ndarray: вектор производных [dw/dx, dθ/dx]
    """
    w, theta = state

    # Вычисляем изгибающий момент
    M_x = calculate_moment(x_coord, q, L)

    # Жесткость с линейной зависимостью от растяжения
    EJ = EJ_0 * (1 + alpha * abs(theta))

    # Система уравнений первого порядка
    dw_dx = theta
    dtheta_dx = M_x / EJ

    return np.array([dw_dx, dtheta_dx])


def runge_kutta_step(state, x_coord, step):
    """
    Один шаг метода Рунге-Кутты 4-го порядка

    Args:
        state (numpy.ndarray): текущее состояние системы [w, θ]
        x_coord (float): текущая координата (м)
        step (float): шаг интегрирования (м)

    Returns:
        numpy.ndarray: состояние системы на следующем шаге [w, θ]
    """
    # Вычисляем коэффициенты k1
    k1 = step * system_equations(state, x_coord)

    # Вычисляем коэффициенты k2 (промежуточная точка)
    k2 = step * system_equations(state + 0.5 * k1, x_coord + 0.5 * step)

    # Вычисляем коэффициенты k3 (другая промежуточная точка)
    k3 = step * system_equations(state + 0.5 * k2, x_coord + 0.5 * step)

    # Вычисляем коэффициенты k4 (конечная точка)
    k4 = step * system_equations(state + k3, x_coord + step)

    # Суммируем взвешенные коэффициенты
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def integrate_beam(initial_theta):
    """
    Интегрирование системы уравнений для заданного начального угла

    Args:
        initial_theta (float): начальный угол поворота (рад)

    Returns:
        tuple: (x_coordinates, solution_trajectory)
    """
    # Начальные условия
    initial_state = np.array([w_0, initial_theta])

    # Инициализация массива решений
    solution = np.zeros((N + 1, 2))
    solution[0] = initial_state

    # Интегрирование методом Рунге-Кутты
    for i in range(N):
        solution[i + 1] = runge_kutta_step(solution[i], x[i], dx)

    return x, solution


def calculate_residual(initial_theta):
    """
    Вычисление невязки граничного условия

    Args:
        initial_theta (float): начальный угол поворота (рад)

    Returns:
        float: невязка (разность между расчетным и заданным прогибом на правом конце)
    """
    _, trajectory = integrate_beam(initial_theta)
    w_final = trajectory[-1, 0]  # прогиб на правом конце
    return w_final - w_L


def shooting_method_regula_falsi(theta_left, theta_right, max_iterations=50, tolerance=1e-8):
    """
    Решение краевой задачи методом стрельбы с коррекцией regula falsi

    Args:
        theta_left (float): левая граница начального угла (рад)
        theta_right (float): правая граница начального угла (рад)
        max_iterations (int): максимальное число итераций
        tolerance (float): требуемая точность

    Returns:
        tuple: (solution_theta, theta_history, residual_history, trajectories)
    """
    # Вычисляем невязки для граничных значений
    residual_left = calculate_residual(theta_left)
    residual_right = calculate_residual(theta_right)

    # История итераций
    theta_history = [theta_left, theta_right]
    residual_history = [residual_left, residual_right]
    trajectories = []

    print("Метод стрельбы с коррекцией regula falsi:")
    print(f"θ_left = {theta_left:.6f} рад, невязка = {residual_left:.6f} м")
    print(f"θ_right = {theta_right:.6f} рад, невязка = {residual_right:.6f} м")

    for iteration in range(max_iterations):
        # Проверка сходимости
        if abs(residual_left) < tolerance:
            print(f"Решение найдено: θ = {theta_left:.6f} рад")
            return theta_left, theta_history, residual_history, trajectories
        if abs(residual_right) < tolerance:
            print(f"Решение найдено: θ = {theta_right:.6f} рад")
            return theta_right, theta_history, residual_history, trajectories

        # Вычисление нового угла методом regula falsi
        theta_new = (residual_right * theta_left - residual_left * theta_right) / (residual_right - residual_left)
        residual_new = calculate_residual(theta_new)

        # Сохраняем траекторию для визуализации
        _, trajectory = integrate_beam(theta_new)
        trajectories.append(trajectory)

        # Сохраняем историю
        theta_history.append(theta_new)
        residual_history.append(residual_new)

        print(f"Итерация {iteration + 1}: θ = {theta_new:.6f} рад, невязка = {residual_new:.6f} м")

        # Обновление границ
        if residual_new * residual_left > 0:
            theta_left = theta_new
            residual_left = residual_new
        else:
            theta_right = theta_new
            residual_right = residual_new

        # Проверка точности
        if abs(residual_new) < tolerance:
            print(f"Решение найдено: θ = {theta_new:.6f} рад")
            return theta_new, theta_history, residual_history, trajectories

    raise ValueError(f"Метод не сошелся за {max_iterations} итераций")


def shooting_method_newton(initial_theta, max_iterations=50, tolerance=1e-8, h=1e-6):
    """
    Решение краевой задачи методом стрельбы с коррекцией Ньютона

    Args:
        initial_theta (float): начальное приближение угла (рад)
        max_iterations (int): максимальное число итераций
        tolerance (float): требуемая точность
        h (float): шаг для численного дифференцирования

    Returns:
        tuple: (solution_theta, theta_history, residual_history, trajectories)
    """
    theta_current = initial_theta
    theta_history = [theta_current]
    residual_history = [calculate_residual(theta_current)]
    trajectories = []

    print("Метод стрельбы с коррекцией Ньютона:")
    print(f"Начальное приближение: θ = {initial_theta:.6f} рад")

    for iteration in range(max_iterations):
        # Вычисление невязки
        residual_current = calculate_residual(theta_current)

        # Проверка сходимости
        if abs(residual_current) < tolerance:
            print(f"Решение найдено: θ = {theta_current:.6f} рад")
            return theta_current, theta_history, residual_history, trajectories

        # Численное вычисление производной
        residual_plus = calculate_residual(theta_current + h)
        residual_minus = calculate_residual(theta_current - h)
        derivative = (residual_plus - residual_minus) / (2 * h)

        # Проверка на нулевую производную
        if abs(derivative) < 1e-12:
            raise ValueError(f"Нулевая производная на итерации {iteration}")

        # Шаг метода Ньютона
        theta_new = theta_current - residual_current / derivative
        residual_new = calculate_residual(theta_new)

        # Сохраняем траекторию для визуализации
        _, trajectory = integrate_beam(theta_new)
        trajectories.append(trajectory)

        # Сохраняем историю
        theta_history.append(theta_new)
        residual_history.append(residual_new)

        print(f"Итерация {iteration + 1}: θ = {theta_new:.6f} рад, невязка = {residual_new:.6f} м")

        theta_current = theta_new

        # Проверка точности
        if abs(residual_new) < tolerance:
            print(f"Решение найдено: θ = {theta_new:.6f} рад")
            return theta_new, theta_history, residual_history, trajectories

    raise ValueError(f"Метод не сошелся за {max_iterations} итераций")


# Основная программа
if __name__ == "__main__":
    print("=" * 60)
    print("МОДЕЛИРОВАНИЕ ИЗГИБА СТЕРЖНЯ С ЛИНЕЙНОЙ ЗАВИСИМОСТЬЮ ЖЕСТКОСТИ")
    print("=" * 60)
    print(f"Длина стержня: L = {L} м")
    print(f"Нагрузка: q = {q} Н/м")
    print(f"Начальная жесткость: EJ = {EJ_0:.2e} Н·м²")
    print(f"Коэффициент зависимости: α = {alpha}")
    print(f"Граничные условия: w(0) = {w_0} м, w(L) = {w_L} м")
    print()

    # Параметры для методов стрельбы
    theta_left_bound = -0.01  # левая граница угла (рад)
    theta_right_bound = 0.01  # правая граница угла (рад)
    theta_initial_guess = 0.005  # начальное приближение для метода Ньютона (рад)

    # Решение методом стрельбы с regula falsi
    print("\n" + "=" * 50)
    theta_solution_rf, theta_hist_rf, residual_hist_rf, trajectories_rf = shooting_method_regula_falsi(
        theta_left_bound, theta_right_bound)

    # Решение методом стрельбы с Ньютоном
    print("\n" + "=" * 50)
    theta_solution_nt, theta_hist_nt, residual_hist_nt, trajectories_nt = shooting_method_newton(
        theta_initial_guess)

    # Сравнение результатов
    print("\n" + "=" * 50)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    print(f"Угол (regula falsi): {theta_solution_rf:.8f} рад")
    print(f"Угол (Ньютон): {theta_solution_nt:.8f} рад")
    print(f"Разница: {abs(theta_solution_rf - theta_solution_nt):.2e} рад")
    print(f"Итераций (regula falsi): {len(theta_hist_rf)}")
    print(f"Итераций (Ньютон): {len(theta_hist_nt)}")

    # Визуализация результатов
    plt.figure(figsize=(15, 10))

    # График 1: Форма изогнутого стержня для финальных решений
    plt.subplot(2, 3, 1)
    x_final_rf, sol_final_rf = integrate_beam(theta_solution_rf)
    x_final_nt, sol_final_nt = integrate_beam(theta_solution_nt)

    plt.plot(x_final_rf, sol_final_rf[:, 0] * 1000, 'b-', linewidth=2, label='Regula Falsi')
    plt.plot(x_final_nt, sol_final_nt[:, 0] * 1000, 'r--', linewidth=2, label='Ньютон')
    plt.xlabel('Координата x (м)')
    plt.ylabel('Прогиб w (мм)')
    plt.title('Форма изогнутого стержня')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График 2: Углы поворота по длине стержня
    plt.subplot(2, 3, 2)
    plt.plot(x_final_rf, sol_final_rf[:, 1], 'b-', linewidth=2, label='Regula Falsi')
    plt.plot(x_final_nt, sol_final_nt[:, 1], 'r--', linewidth=2, label='Ньютон')
    plt.xlabel('Координата x (м)')
    plt.ylabel('Угол поворота θ (рад)')
    plt.title('Углы поворота по длине стержня')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График 3: Сходимость метода regula falsi
    plt.subplot(2, 3, 3)
    plt.plot(range(len(residual_hist_rf)), np.abs(residual_hist_rf), 'bo-', alpha=0.7)
    plt.xlabel('Номер итерации')
    plt.ylabel('Невязка (м)')
    plt.title('Сходимость метода Regula Falsi')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # График 4: Сходимость метода Ньютона
    plt.subplot(2, 3, 4)
    plt.plot(range(len(residual_hist_nt)), np.abs(residual_hist_nt), 'ro-', alpha=0.7)
    plt.xlabel('Номер итерации')
    plt.ylabel('Невязка (м)')
    plt.title('Сходимость метода Ньютона')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # График 5: Траектории итераций для regula falsi
    plt.subplot(2, 3, 5)
    for i, trajectory in enumerate(trajectories_rf[:6]):  # первые 6 итераций
        w_values = trajectory[:, 0] * 1000  # переводим в мм
        if i == len(trajectories_rf[:6]) - 1:
            plt.plot(x, w_values, 'b-', linewidth=2, label=f'Итерация {i + 1}')
        else:
            plt.plot(x, w_values, 'b--', alpha=0.5, label=f'Итерация {i + 1}')
    plt.xlabel('Координата x (м)')
    plt.ylabel('Прогиб w (мм)')
    plt.title('Траектории итераций (Regula Falsi)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График 6: Траектории итераций для Ньютона
    plt.subplot(2, 3, 6)
    for i, trajectory in enumerate(trajectories_nt[:6]):  # первые 6 итераций
        w_values = trajectory[:, 0] * 1000  # переводим в мм
        if i == len(trajectories_nt[:6]) - 1:
            plt.plot(x, w_values, 'r-', linewidth=2, label=f'Итерация {i + 1}')
        else:
            plt.plot(x, w_values, 'r--', alpha=0.5, label=f'Итерация {i + 1}')
    plt.xlabel('Координата x (м)')
    plt.ylabel('Прогиб w (мм)')
    plt.title('Траектории итераций (Ньютон)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Анализ распределения моментов и напряжений
    print("\n" + "=" * 50)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ МОМЕНТОВ И НАПРЯЖЕНИЙ:")

    # Вычисляем моменты и напряжения для финального решения
    x_final, sol_final = integrate_beam(theta_solution_rf)
    moments = np.array([calculate_moment(x_i, q, L) for x_i in x_final])
    angles = sol_final[:, 1]
    EJ_values = EJ_0 * (1 + alpha * np.abs(angles))
    curvatures = moments / EJ_values

    print(f"Максимальный изгибающий момент: {np.max(np.abs(moments)):.2f} Н·м")
    print(f"Максимальный прогиб: {np.max(np.abs(sol_final[:, 0])) * 1000:.2f} мм")
    print(f"Максимальный угол поворота: {np.max(np.abs(angles)):.4f} рад")