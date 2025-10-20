import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import trapezoid

# ПАРАМЕТРЫ ЗАДАЧИ
L = 1.0  # длина стержня
k0 = 1.0  # базовая теплопроводность
t_end = 0.5  # конечное время
x_points_list = [51, 101, 151, 201]  # различные сетки для анализа сходимости

# АНАЛИТИЧЕСКОЕ РЕШЕНИЕ ДЛЯ ПОСТОЯННОЙ ТЕПЛОПРОВОДНОСТИ (k = k0)
def analytical_solution_constant_k(x, t, k0, L, n_terms=100):
    u = np.ones_like(x)  # граничное условие u(0,t) = 1
    for n in range(1, n_terms + 1):
        lambda_n = (2 * n - 1) * np.pi / (2 * L)
        coeff = 2 / (lambda_n * L)
        u -= coeff * np.sin(lambda_n * x) * np.exp(-k0 * lambda_n**2 * t)
    return u

# ФУНКЦИЯ ПЕРЕМЕННОЙ ТЕПЛОПРОВОДНОСТИ
def k_func(x, k0):
    return k0 * (1 + 0.5 * np.sin(2 * np.pi * x))

# ЯВНАЯ СХЕМА FTCS (С УЧЕТОМ CFL УСЛОВИЯ)
def ftcs_solver(Nx, t, L, k0):
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)
    k = k_func(x, k0)
    max_k = np.max(k)
    dt_stable = 0.5 * dx**2 / max_k      # Автоматический подбор шага по времени для устойчивости (CFL ≤ 0.5)
    Nt = max(1, int(t / dt_stable)) + 1
    dt_actual = t / Nt
    u = np.zeros(Nx)  # начальное условие u(x,0) = 0
    u[0] = 1.0  # граничное условие u(0,t) = 1
    alpha = dt_actual / dx**2 # Коэффициент для FTCS схемы

    for n in range(Nt):
        u_new = u.copy()
        for i in range(1, Nx - 1):
            k_plus = 0.5 * (k[i] + k[i + 1])
            k_minus = 0.5 * (k[i] + k[i - 1])
            flux_plus = k_plus * (u[i + 1] - u[i])
            flux_minus = k_minus * (u[i] - u[i - 1])
            u_new[i] = u[i] + alpha * (flux_plus - flux_minus)
        u_new[0] = 1.0  # u(0,t) = 1
        u_new[-1] = u_new[-2]  # нулевой поток на правой границе (условие Неймана)
        u = u_new
    return x, u

# СХЕМА КРАНКА-НИКОЛЬСОНА
def crank_nicolson_solver(Nx, t, L, k0):
    dx = L / (Nx - 1)
    dt = min(0.01, t/10)  # Для неявной схемы можно использовать больший шаг по времени
    Nt = max(1, int(t / dt))
    dt_actual = t / Nt
    x = np.linspace(0, L, Nx)
    u = np.zeros(Nx)  # начальное условие
    u[0] = 1.0  # граничное условие
    k = k_func(x, k0)
    alpha = dt_actual / (2 * dx**2) # Коэффициенты для матричной системы
    main_diag = np.ones(Nx)
    upper_diag = np.zeros(Nx - 1)
    lower_diag = np.zeros(Nx - 1)
    for i in range(1, Nx - 1):
        k_plus = 0.5 * (k[i] + k[i + 1])
        k_minus = 0.5 * (k[i] + k[i - 1])
        main_diag[i] = 1 + alpha * (k_plus + k_minus)
        upper_diag[i] = -alpha * k_plus
        lower_diag[i - 1] = -alpha * k_minus
    # Граничные условия
    main_diag[0] = 1.0
    upper_diag[0] = 0.0
    main_diag[-1] = 1.0
    lower_diag[-1] = -1.0  # для условия Неймана
    A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
    for n in range(Nt):
        b = u.copy()
        for i in range(1, Nx - 1):
            k_plus = 0.5 * (k[i] + k[i + 1])
            k_minus = 0.5 * (k[i] + k[i - 1])
            flux_plus = k_plus * (u[i + 1] - u[i])
            flux_minus = k_minus * (u[i] - u[i - 1])
            b[i] = u[i] + alpha * (flux_plus - flux_minus)
        # Граничные условия в правой части
        b[0] = 1.0
        b[-1] = 0.0  # для условия Неймана
        u = spsolve(A, b) 
    return x, u

# РАСЧЕТ ТЕПЛОВОЙ ЭНЕРГИИ (интеграл от u по x)
def calculate_thermal_energy(u, dx):
    return trapezoid(u, dx=dx)

# АНАЛИЗ СХОДИМОСТИ
def convergence_analysis():
    errors_ftcs = []
    errors_cn = []
    dx_values = []
    t_test = 0.1  
    Nx_fine = 501
    x_fine, u_fine = crank_nicolson_solver(Nx_fine, t_test, L, k0)
    
    for Nx in x_points_list:
        dx = L / (Nx - 1)
        x_ftcs, u_ftcs = ftcs_solver(Nx, t_test, L, k0)
        x_cn, u_cn = crank_nicolson_solver(Nx, t_test, L, k0)
        u_fine_on_ftcs = np.interp(x_ftcs, x_fine, u_fine)
        u_fine_on_cn = np.interp(x_cn, x_fine, u_fine)
        error_ftcs = np.sqrt(dx * np.sum((u_ftcs - u_fine_on_ftcs)**2))
        error_cn = np.sqrt(dx * np.sum((u_cn - u_fine_on_cn)**2))
        errors_ftcs.append(error_ftcs)
        errors_cn.append(error_cn)
        dx_values.append(dx)
    
    return dx_values, errors_ftcs, errors_cn

Nx = 101
time_points = [0.01, 0.1, 0.5]
    
plt.figure(figsize=(15, 10))
for i, t in enumerate(time_points):
    x_ftcs, u_ftcs = ftcs_solver(Nx, t, L, k0)
    x_cn, u_cn = crank_nicolson_solver(Nx, t, L, k0)
    u_analytical = analytical_solution_constant_k(x_ftcs, t, k0, L)
        
    plt.subplot(2, 2, i + 1)
    plt.plot(x_ftcs, u_ftcs, 'b-', linewidth=2, label='FTCS')
    plt.plot(x_cn, u_cn, 'r--', linewidth=2, label='Кранк-Никольсон')
    plt.plot(x_ftcs, u_analytical, 'g:', linewidth=2, label='Аналитическое (k=const)')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'Температура в момент t = {t}')
    plt.legend()
    plt.grid(True)
    
plt.subplot(2, 2, 4)
    
time_array = np.linspace(0.001, 0.5, 20)  
energy_ftcs = []
energy_cn = []
energy_analytical = []
    
for j, t in enumerate(time_array):
    x_ftcs, u_ftcs = ftcs_solver(Nx, t, L, k0)
    x_cn, u_cn = crank_nicolson_solver(Nx, t, L, k0)
    u_analytical = analytical_solution_constant_k(x_ftcs, t, k0, L)
    dx = L / (Nx - 1)
    energy_ftcs.append(calculate_thermal_energy(u_ftcs, dx))
    energy_cn.append(calculate_thermal_energy(u_cn, dx))
    energy_analytical.append(calculate_thermal_energy(u_analytical, dx))
    
plt.plot(time_array, energy_ftcs, 'b-', label='FTCS')
plt.plot(time_array, energy_cn, 'r--', label='Кранк-Никольсон')
plt.plot(time_array, energy_analytical, 'g:', label='Аналитическое (k=const)')
plt.xlabel('Время')
plt.ylabel('Тепловая энергия')
plt.title('Изменение тепловой энергии системы')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
    
dx_values, errors_ftcs, errors_cn = convergence_analysis()

plt.figure(figsize=(10, 6))
plt.loglog(dx_values, errors_ftcs, 'bo-', linewidth=2, markersize=8, label='FTCS')
plt.loglog(dx_values, errors_cn, 'rs-', linewidth=2, markersize=8, label='Кранк-Никольсон')    
dx_theory = np.array(dx_values)
plt.loglog(dx_theory, 0.1 * dx_theory, 'k--', label='~Δx (первый порядок)')
plt.loglog(dx_theory, 0.01 * dx_theory**2, 'k:', label='~Δx² (второй порядок)')
plt.xlabel('Δx')
plt.ylabel('Ошибка (L2 норма)')
plt.title('Анализ сходимости численных схем')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.show()
