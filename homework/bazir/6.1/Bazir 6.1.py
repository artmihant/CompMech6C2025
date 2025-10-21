import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Параметры задачи
L = 1.0
k0 = 1.0
T_max = 0.5
times = [0.01, 0.1, 0.5]

# Явная схема FTCS с автоматическим выбором шага
def ftcs_solve(Nx, T_final):
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)
    k = k0 * (1 + 0.5 * np.sin(2 * np.pi * x))
    
    # Автоматический выбор dt по условию устойчивости CFL
    k_max = np.max(k)
    dt = 0.3 * dx**2 / (2 * k_max)
    Nt = max(1, int(T_final / dt))
    dt = T_final / Nt
    u = np.zeros(Nx)
    u[0] = 1.0  
    k_half = 0.5 * (k[1:] + k[:-1])
    for step in range(Nt):
        u_new = u.copy()
        for i in range(1, Nx-1):
            dudx_right = (u[i+1] - u[i]) / dx
            dudx_left = (u[i] - u[i-1]) / dx
            u_new[i] = u[i] + dt/dx * (k_half[i] * dudx_right - k_half[i-1] * dudx_left)
        u_new[0] = 1.0
        u_new[-1] = u_new[-2]
        u = u_new
    return x, u

# Схема Кранка-Николсона
def crank_nicolson_solve(Nx, T_final):
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)
    k = k0 * (1 + 0.5 * np.sin(2 * np.pi * x))
    dt = 0.001
    Nt = max(1, int(T_final / dt))
    dt = T_final / Nt
    u = np.zeros(Nx)
    u[0] = 1.0
    k_half = 0.5 * (k[1:] + k[:-1])
    alpha = dt / (2 * dx**2)
    main_diag = np.ones(Nx)
    lower_diag = np.zeros(Nx-1)
    upper_diag = np.zeros(Nx-1)
    main_diag[1:-1] = 1 + alpha * (k_half[1:] + k_half[:-1])
    lower_diag[1:] = -alpha * k_half[:-1]
    upper_diag[:-1] = -alpha * k_half[1:]
    main_diag[0] = 1.0
    upper_diag[0] = 0.0
    main_diag[-1] = 1.0
    lower_diag[-1] = -1.0
    A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
    for step in range(Nt):
        b = np.zeros(Nx)
        b[1:-1] = u[1:-1] + alpha * (
            k_half[1:] * (u[2:] - u[1:-1]) - 
            k_half[:-1] * (u[1:-1] - u[:-2])
        )
        b[0] = 1.0
        b[-1] = 0.0
        u = spsolve(A, b)
    return x, u

# Построение графиков температуры
Nx = 100
plt.figure(figsize=(12, 8))

for i, t in enumerate(times):
    plt.subplot(2, 2, i+1)
    x_ftcs, u_ftcs = ftcs_solve(Nx, t)
    x_cn, u_cn = crank_nicolson_solve(Nx, t)
    plt.plot(x_ftcs, u_ftcs, 'r--', label='FTCS')
    plt.plot(x_cn, u_cn, 'b-.', label='Кранк-Николсон')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f't = {t}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# График тепловой энергии
def calculate_energy(u, dx):
    return np.trapezoid(u, dx=dx)

time_points = np.linspace(0.001, T_max, 20)
energy_ftcs = []
energy_cn = []

for t in time_points:
    _, u_ftcs = ftcs_solve(50, t)
    _, u_cn = crank_nicolson_solve(50, t)
    dx = L / 49
    energy_ftcs.append(calculate_energy(u_ftcs, dx))
    energy_cn.append(calculate_energy(u_cn, dx))

plt.figure(figsize=(10, 6))
plt.plot(time_points, energy_ftcs, 'ro-', label='FTCS', markersize=4)
plt.plot(time_points, energy_cn, 'bs-', label='Кранк-Николсон', markersize=4)
plt.xlabel('Время')
plt.ylabel('Тепловая энергия')
plt.legend()
plt.title('Изменение тепловой энергии системы')
plt.grid(True, alpha=0.3)
plt.show()

# Оценка порядка сходимости
Nx_values = [25, 50, 100, 200]
errors_ftcs = []
errors_cn = []
T_conv = 0.1

# Эталонное решение на мелкой сетке
print("Calculating...")
Nx_ref = 400
x_ref, u_ref_ftcs = ftcs_solve(Nx_ref, T_conv)
_, u_ref_cn = crank_nicolson_solve(Nx_ref, T_conv)

for Nx in Nx_values:
    x, u_ftcs = ftcs_solve(Nx, T_conv)
    x, u_cn = crank_nicolson_solve(Nx, T_conv)
    u_ref_ftcs_interp = np.interp(x, x_ref, u_ref_ftcs)
    u_ref_cn_interp = np.interp(x, x_ref, u_ref_cn)
    error_ftcs = np.sqrt(np.sum((u_ftcs - u_ref_ftcs_interp)**2) / Nx)
    error_cn = np.sqrt(np.sum((u_cn - u_ref_cn_interp)**2) / Nx)
    errors_ftcs.append(error_ftcs)
    errors_cn.append(error_cn)

dx_values = [L / (Nx - 1) for Nx in Nx_values]

plt.figure(figsize=(10, 6))
plt.loglog(dx_values, errors_ftcs, 'ro-', label='FTCS', linewidth=2, markersize=8)
plt.loglog(dx_values, errors_cn, 'bs-', label='Кранк-Николсон', linewidth=2, markersize=8)

for i in range(1, len(Nx_values)):
    p_ftcs = np.log(errors_ftcs[i-1] / errors_ftcs[i]) / np.log(dx_values[i-1] / dx_values[i])
    p_cn = np.log(errors_cn[i-1] / errors_cn[i]) / np.log(dx_values[i-1] / dx_values[i])
    print(f"FTCS: при переходе от Nx={Nx_values[i-1]} к Nx={Nx_values[i]}: {p_ftcs:.3f}")
    print(f"Кранк-Николсон: при переходе от Nx={Nx_values[i-1]} к Nx={Nx_values[i]}: {p_cn:.3f}")

# Теоретическая линия O(Δx²)
ref_line = [errors_ftcs[0] * (dx/dx_values[0])**2 for dx in dx_values]
plt.loglog(dx_values, ref_line, 'k--', label='Теоретический O(Δx²)', alpha=0.7)

plt.xlabel('Δx')
plt.ylabel('Среднеквадратичная ошибка')
plt.legend()
plt.title('Оценка порядка сходимости (относительно решения на мелкой сетке)')
plt.grid(True, alpha=0.3)
plt.show()