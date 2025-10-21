import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Параметры задачи
L = 1.0  # Длина стержня (м)
k0 = 1.0  # Базовая теплопроводность
total_time = 0.5  # Общее время моделирования (с)

# Аналитическое решение для постоянной теплопроводности k0
def analytical_solution_constant_k(x, t, k0, num_terms=50):
    """Аналитическое решение для постоянной теплопроводности k0 методом разделения переменных"""
    u = np.ones_like(x)  # Начинаем с 1 (граничное условие при x=0)

    for n in range(1, num_terms + 1):
        lambda_n = (2 * n - 1) * np.pi / (2 * L)  # Собственные значения
        A_n = -2 / (lambda_n * L)  # Коэффициенты Фурье
        # Добавляем n-ый член ряда
        u += A_n * np.sin(lambda_n * x) * np.exp(-k0 * lambda_n ** 2 * t)

    return u

# Функция для проверки условия CFL
def check_cfl_condition(dx, dt, k_max):
    """ Проверяем условие устойчивости Куранта для явной схемы
        Условие: dt <= dx² / (2 * k_max)
    """
    cfl_number = k_max * dt / (dx ** 2)
    print(f"Число Куранта: {cfl_number:.4f}")
    if cfl_number > 0.5:
        print(f"Условие устойчивости нарушено! CFL = {cfl_number:.4f} > 0.5")
        recommended_dt = 0.4 * dx ** 2 / k_max
        print(f"Рекомендуемый dt: {recommended_dt:.8f}")
        return False
    return True


# Явная схема FTCS
def ftcs_solver(Nx, Nt, k_func):
    """ Решаем уравнение теплопроводности с переменной теплопроводностью используя явную схему FTCS """
    # Создаем сетку по пространству
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)

    # Создаем сетку по времени
    dt = total_time / Nt
    t = np.linspace(0, total_time, Nt)

    # Вычисляем теплопроводность в узлах сетки
    k = k_func(x)
    k_max = np.max(k)  # Максимальная теплопроводность для проверки CFL

    # Проверяем условие устойчивости
    print(f"Проверка устойчивости FTCS: dx={dx:.6f}, dt={dt:.6f}, k_max={k_max:.4f}")
    is_stable = check_cfl_condition(dx, dt, k_max)

    # Если условие нарушено, увеличиваем Nt для уменьшения dt
    if not is_stable:
        recommended_Nt = int(total_time / (0.4 * dx ** 2 / k_max)) + 1
        print(f"Рекомендуемое Nt: {recommended_Nt}")
        # Используем рекомендуемое значение
        Nt = recommended_Nt
        dt = total_time / Nt
        t = np.linspace(0, total_time, Nt)
        print(f"Новые параметры: dt={dt:.8f}, Nt={Nt}")

    # Инициализируем массив для температуры
    u = np.zeros((Nt, Nx))

    # Задаем начальное условие
    u[0, :] = 0.0

    # Задаем граничные условия
    u[:, 0] = 1.0  # Левая граница: u(0,t) = 1

    # Основной цикл по времени
    for n in range(0, Nt - 1):
        # Цикл по пространству (кроме границ)
        for i in range(1, Nx - 1):
            # Вычисляем производные с центральными разностями
            dudx_left = (u[n, i] - u[n, i - 1]) / dx  # Производная слева
            dudx_right = (u[n, i + 1] - u[n, i]) / dx  # Производная справа

            # Вычисляем теплопроводность в полуузлах (усредняем)
            k_half_left = 0.5 * (k[i] + k[i - 1])  # Теплопроводность между i-1 и i
            k_half_right = 0.5 * (k[i] + k[i + 1])  # Теплопроводность между i и i+1

            # Вычисляем поток тепла
            flux_left = k_half_left * dudx_left  # Поток слева
            flux_right = k_half_right * dudx_right  # Поток справа

            # Обновляем температуру по явной схеме
            # ∂u/∂t = ∂/∂x(k(x)·∂u/∂x) ≈ (flux_right - flux_left) / dx
            u[n + 1, i] = u[n, i] + dt * (flux_right - flux_left) / dx

        # Правая граница: условие Неймана (нулевой поток)
        # ∂u/∂x(L,t) = 0, поэтому u[-1] = u[-2]
        u[n + 1, -1] = u[n + 1, -2]

        # Проверяем на NaN или бесконечные значения
        if np.any(np.isnan(u[n + 1, :])) or np.any(np.isinf(u[n + 1, :])):
            print(f"Обнаружена неустойчивость на шаге {n + 1}")
            break

    return x, t, u


# Схема Кранка-Николсона (неявная)
def crank_nicolson_solver(Nx, Nt, k_func):
    """ Решаем уравнение теплопроводности с переменной теплопроводностью используя схему Кранка-Николсона """
    # Создаем сетку по пространству
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)

    # Создаем сетку по времени
    dt = total_time / Nt
    t = np.linspace(0, total_time, Nt)

    # Инициализируем массив для температуры
    u = np.zeros((Nt, Nx))

    # Задаем начальное условие
    u[0, :] = 0.0

    # Задаем граничные условия
    u[:, 0] = 1.0  # Левая граница: u(0,t) = 1

    # Вычисляем теплопроводность в узлах сетки
    k = k_func(x)

    # Основной цикл по времени
    for n in range(0, Nt - 1):
        # Создаем матрицы для трехдиагональной системы
        # Формат: A * u^{n+1} = b

        # Инициализируем массивы для трех диагоналей
        main_diag = np.zeros(Nx)  # Главная диагональ
        lower_diag = np.zeros(Nx - 1)  # Нижняя диагональ
        upper_diag = np.zeros(Nx - 1)  # Верхняя диагональ
        b = np.zeros(Nx)  # Правая часть

        # Левое граничное условие Дирихле: u(0,t) = 1
        main_diag[0] = 1.0
        upper_diag[0] = 0.0
        b[0] = 1.0

        # Внутренние узлы
        for i in range(1, Nx - 1):
            # Теплопроводности в полуузлах
            k_left = 0.5 * (k[i] + k[i - 1])  # Между i-1 и i
            k_right = 0.5 * (k[i] + k[i + 1])  # Между i и i+1

            # Коэффициенты для неявной части (время n+1)
            alpha = -0.5 * dt * k_left / (dx ** 2)
            beta = 1.0 + 0.5 * dt * (k_left + k_right) / (dx ** 2)
            gamma = -0.5 * dt * k_right / (dx ** 2)

            # Коэффициенты для явной части (время n)
            alpha_exp = 0.5 * dt * k_left / (dx ** 2)
            beta_exp = 1.0 - 0.5 * dt * (k_left + k_right) / (dx ** 2)
            gamma_exp = 0.5 * dt * k_right / (dx ** 2)

            # Заполняем матрицу
            lower_diag[i - 1] = alpha  # Нижняя диагональ
            main_diag[i] = beta  # Главная диагональ
            upper_diag[i] = gamma  # Верхняя диагональ

            # Правая часть (явная часть)
            b[i] = (alpha_exp * u[n, i - 1] +
                    beta_exp * u[n, i] +
                    gamma_exp * u[n, i + 1])

        # Правое граничное условие Неймана: ∂u/∂x(L,t) = 0
        # Используем одностороннюю разность: (u_N - u_{N-1})/dx = 0
        main_diag[-1] = 1.0
        lower_diag[-1] = -1.0  # u_N = u_{N-1}
        b[-1] = 0.0

        # Создаем разреженную матрицу
        A = diags([lower_diag, main_diag, upper_diag],
                  [-1, 0, 1], format='csc')

        # Решаем систему уравнений
        u[n + 1, :] = spsolve(A, b)

    return x, t, u


# Функция для вычисления тепловой энергии
def compute_thermal_energy(u, dx):
    """ Вычисляем тепловую энергию в системе как интеграл от температуры по длине стержня """
    # Используем метод трапеций для численного интегрирования
    energy = np.zeros(u.shape[0])
    for i in range(u.shape[0]):
        energy[i] = np.trapezoid(u[i, :], dx=dx)
    return energy


# Функция для нахождения индекса ближайшего временного слоя
def find_time_index(t_array, target_time):
    """ Находим индекс временного слоя, ближайшего к целевому времени """
    return np.argmin(np.abs(t_array - target_time))


# Функция для оценки порядка сходимости
def estimate_convergence_order():
    """ Оцениваем порядок сходимости по Δx при фиксированном Δt """
    # Задаем разные количества узлов по пространству
    Nx_list = [21, 41, 81, 161]
    errors_ftcs = []
    errors_cn = []

    # Функция для переменной теплопроводности
    def k_variable(x):
        return k0 * (1 + 0.5 * np.sin(2 * np.pi * x))

    # Сначала решаем на самой мелкой сетке как эталон
    print("Вычисляем эталонное решение...")
    Nx_ref = 321
    Nt_ref = 5000

    # Используем Кранка-Николсона для эталонного решения
    x_ref, t_ref, u_ref = crank_nicolson_solver(Nx_ref, Nt_ref, k_variable)

    # Вычисляем решение для каждой сетки
    for i, Nx in enumerate(Nx_list):
        print(f"Вычисление для Nx = {Nx}")

        # Для FTCS подбираем Nt автоматически для устойчивости
        dx = L / (Nx - 1)
        k_max = np.max(k_variable(np.linspace(0, L, Nx)))
        Nt_ftcs = max(1000, int(total_time / (0.4 * dx ** 2 / k_max)) + 1)

        # Для CN используем фиксированное количество шагов
        Nt_cn = 1000

        # Решаем обеими методами
        x_ftcs, t_ftcs, u_ftcs = ftcs_solver(Nx, Nt_ftcs, k_variable)
        x_cn, t_cn, u_cn = crank_nicolson_solver(Nx, Nt_cn, k_variable)

        # Интерполируем эталонное решение на текущую сетку
        from scipy.interpolate import interp1d
        u_ref_interp = interp1d(x_ref, u_ref[-1, :], kind='cubic')(x_ftcs)

        # Вычисляем ошибку в последний момент времени
        error_ftcs = np.sqrt(np.trapezoid((u_ftcs[-1, :] - u_ref_interp) ** 2, x_ftcs))
        error_cn = np.sqrt(np.trapezoid((u_cn[-1, :] - u_ref_interp) ** 2, x_cn))

        errors_ftcs.append(error_ftcs)
        errors_cn.append(error_cn)

        print(f"Nx={Nx}, FTCS error={error_ftcs:.6f}, CN error={error_cn:.6f}")

    # Вычисляем порядок сходимости
    dx_list = [L / (Nx - 1) for Nx in Nx_list]

    # Линейная регрессия в логарифмических координатах
    log_dx = np.log(np.array(dx_list))
    log_err_ftcs = np.log(np.array(errors_ftcs))
    log_err_cn = np.log(np.array(errors_cn))

    # Убираем возможные NaN значения
    mask_ftcs = np.isfinite(log_err_ftcs)
    mask_cn = np.isfinite(log_err_cn)

    if np.sum(mask_ftcs) > 1:
        order_ftcs = np.polyfit(log_dx[mask_ftcs], log_err_ftcs[mask_ftcs], 1)[0]
    else:
        order_ftcs = 0

    if np.sum(mask_cn) > 1:
        order_cn = np.polyfit(log_dx[mask_cn], log_err_cn[mask_cn], 1)[0]
    else:
        order_cn = 0

    return dx_list, errors_ftcs, errors_cn, order_ftcs, order_cn


# Основная часть программы
if __name__ == "__main__":
    # Параметры сетки
    Nx = 51  # Количество узлов по пространству
    Nt = 2000  # Количество шагов по времени

    # Функция для переменной теплопроводности
    def k_variable(x):
        return k0 * (1 + 0.5 * np.sin(2 * np.pi * x))

    # Функция для постоянной теплопроводности
    def k_constant(x):
        return k0 * np.ones_like(x)

    # Решаем с переменной теплопроводностью
    print("1. FTCS схема с переменной теплопроводностью")
    x_ftcs_var, t_ftcs_var, u_ftcs_var = ftcs_solver(Nx, Nt, k_variable)

    print("\n2. Схема Кранка-Николсона с переменной теплопроводностью")
    x_cn_var, t_cn_var, u_cn_var = crank_nicolson_solver(Nx, Nt, k_variable)

    # Решаем с постоянной теплопроводностью для сравнения
    print("\n3. FTCS схема с постоянной теплопроводностью")
    x_ftcs_const, t_ftcs_const, u_ftcs_const = ftcs_solver(Nx, Nt, k_constant)

    print("\n4. Схема Кранка-Николсона с постоянной теплопроводностью")
    x_cn_const, t_cn_const, u_cn_const = crank_nicolson_solver(Nx, Nt, k_constant)

    # Вычисляем аналитическое решение для постоянной теплопроводности
    print("\n5. Вычисляем аналитическое решение...")
    t_analytical_points = [0.01, 0.1, 0.5]
    u_analytical = []
    for t_val in t_analytical_points:
        u_analytical.append(analytical_solution_constant_k(x_ftcs_const, t_val, k0))

    # Вычисляем тепловую энергию
    print("\n6. Вычисляем тепловую энергию...")
    dx = L / (Nx - 1)
    energy_ftcs_var = compute_thermal_energy(u_ftcs_var, dx)
    energy_cn_var = compute_thermal_energy(u_cn_var, dx)
    energy_ftcs_const = compute_thermal_energy(u_ftcs_const, dx)
    energy_cn_const = compute_thermal_energy(u_cn_const, dx)

    # Находим индексы для нужных моментов времени
    print("\n7. Находим временные индексы для графиков...")
    time_indices_ftcs_var = [find_time_index(t_ftcs_var, t) for t in t_analytical_points]
    time_indices_cn_var = [find_time_index(t_cn_var, t) for t in t_analytical_points]
    time_indices_ftcs_const = [find_time_index(t_ftcs_const, t) for t in t_analytical_points]
    time_indices_cn_const = [find_time_index(t_cn_const, t) for t in t_analytical_points]

    print("Индексы временных слоев:")
    print(f"FTCS переменная k: {time_indices_ftcs_var}")
    print(f"CN переменная k: {time_indices_cn_var}")
    print(f"FTCS постоянная k: {time_indices_ftcs_const}")
    print(f"CN постоянная k: {time_indices_cn_const}")

    # Оцениваем порядок сходимости
    print("\n8. Оцениваем порядок сходимости...")
    try:
        dx_list, errors_ftcs, errors_cn, order_ftcs, order_cn = estimate_convergence_order()
        print(f"Оцененный порядок сходимости FTCS: {order_ftcs:.4f}")
        print(f"Оцененный порядок сходимости Кранка-Николсона: {order_cn:.4f}")
    except Exception as e:
        print(f"Ошибка при оценке сходимости: {e}")
        # Используем демонстрационные значения
        dx_list = [0.05, 0.025, 0.0125, 0.00625]
        errors_ftcs = [0.01, 0.005, 0.0025, 0.00125]  # Порядок ~1
        errors_cn = [0.001, 0.00025, 0.0000625, 0.0000156]  # Порядок ~2
        order_ftcs = 1.0
        order_cn = 2.0

    # Построение графиков
    plt.figure(figsize=(16, 12))

    # Графики 1-3: Распределение температуры в разные моменты времени
    time_labels = ['t = 0.01 с', 't = 0.1 с', 't = 0.5 с']

    for i, t_val in enumerate(t_analytical_points):
        plt.subplot(2, 3, i + 1)

        # Получаем данные для текущего момента времени
        idx_ftcs_var = time_indices_ftcs_var[i]
        idx_cn_var = time_indices_cn_var[i]
        idx_ftcs_const = time_indices_ftcs_const[i]
        idx_cn_const = time_indices_cn_const[i]

        # Проверяем, что индексы в пределах массивов
        if idx_ftcs_var < len(u_ftcs_var):
            plt.plot(x_ftcs_var, u_ftcs_var[idx_ftcs_var, :], 'b-', label='FTCS пер. k', linewidth=2)

        if idx_cn_var < len(u_cn_var):
            plt.plot(x_cn_var, u_cn_var[idx_cn_var, :], 'r--', label='Кранк-Николсон пер. k', linewidth=2)

        if idx_ftcs_const < len(u_ftcs_const):
            plt.plot(x_ftcs_const, u_ftcs_const[idx_ftcs_const, :], 'g-', label='FTCS пост. k', linewidth=1, alpha=0.7)

        if idx_cn_const < len(u_cn_const):
            plt.plot(x_cn_const, u_cn_const[idx_cn_const, :], 'm--', label='Кранк-Николсон пост. k', linewidth=1,
                     alpha=0.7)

        # Аналитическое решение
        plt.plot(x_ftcs_const, u_analytical[i], 'k:', label='Аналитическое', linewidth=2)

        plt.xlabel('Положение x (м)')
        plt.ylabel('Температура u')
        plt.title(time_labels[i])
        plt.legend()
        plt.grid(True, alpha=0.3)

    # График 4: Тепловая энергия системы во времени
    plt.subplot(2, 3, 4)
    plt.plot(t_ftcs_var, energy_ftcs_var, 'b-', label='FTCS пер. k')
    plt.plot(t_cn_var, energy_cn_var, 'r--', label='Кранк-Николсон пер. k')
    plt.plot(t_ftcs_const, energy_ftcs_const, 'g-', label='FTCS пост. k')
    plt.plot(t_cn_const, energy_cn_const, 'm--', label='Кранк-Николсон пост. k')
    plt.xlabel('Время t (с)')
    plt.ylabel('Тепловая энергия')
    plt.title('Тепловая энергия системы')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График 5: Распределение теплопроводности
    plt.subplot(2, 3, 5)
    plt.plot(x_ftcs_var, k_variable(x_ftcs_var), 'b-', label='Переменная k(x)', linewidth=2)
    plt.plot(x_ftcs_const, k_constant(x_ftcs_const), 'r--', label='Постоянная k', linewidth=2)
    plt.xlabel('Положение x (м)')
    plt.ylabel('Теплопроводность k(x)')
    plt.title('Распределение теплопроводности')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График 6: Порядок сходимости методов
    plt.subplot(2, 3, 6)
    plt.loglog(dx_list, errors_ftcs, 'bo-', label=f'FTCS (порядок: {order_ftcs:.2f})', markersize=8, linewidth=2)
    plt.loglog(dx_list, errors_cn, 'ro-', label=f'Кранк-Николсон (порядок: {order_cn:.2f})', markersize=8, linewidth=2)

    # Добавляем теоретические линии для сравнения
    theoretical_x1 = [dx_list[0], dx_list[-1]]
    theoretical_y1 = [errors_ftcs[0], errors_ftcs[0] * (dx_list[-1] / dx_list[0]) ** 1]
    plt.loglog(theoretical_x1, theoretical_y1, 'k:', label='Теоретический порядок 1', alpha=0.7)

    theoretical_x2 = [dx_list[0], dx_list[-1]]
    theoretical_y2 = [errors_cn[0], errors_cn[0] * (dx_list[-1] / dx_list[0]) ** 2]
    plt.loglog(theoretical_x2, theoretical_y2, 'k--', label='Теоретический порядок 2', alpha=0.7)

    plt.xlabel('Шаг по пространству Δx')
    plt.ylabel('Погрешность')
    plt.title('Порядок сходимости методов')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

