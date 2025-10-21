import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import erfc

L = 1.0
Nx = 201
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)

k0 = 0.01
k = k0 * (1 + 0.5 * np.sin(2 * np.pi * x))

dt = 0.0005
t_total = 0.5
Nt = int(t_total / dt)

def analytical_solution_constant_k(x, t, k0, L, N_terms=100):
    if t == 0:
        return np.zeros_like(x)
    
    u = np.ones_like(x) 
    
    for n in range(1, N_terms):
        lambda_n = (2*n - 1) * np.pi / (2*L)
        coef = 2 / ((2*n - 1) * np.pi)
        u -= coef * np.sin(lambda_n * x) * np.exp(-k0 * lambda_n**2 * t)
    
    return u

u0 = np.zeros(Nx)
u0[0] = 1.0

def ftcs_variable_k(u0, k, dx, dt, Nt, save_all_steps=True):
    Nx = len(u0)
    u = u0.copy()
    if save_all_steps:
        u_hist = [u.copy()]
    for n in range(Nt):
        u_new = u.copy()
        for i in range(1, Nx - 1):
            k_half_right = 0.5 * (k[i] + k[i + 1])
            k_half_left = 0.5 * (k[i] + k[i - 1])
            u_new[i] = u[i] + dt / dx**2 * (
                k_half_right * (u[i + 1] - u[i]) - k_half_left * (u[i] - u[i - 1])
            )
        u_new[0] = 1.0
        u_new[-1] = u_new[-2]
        u = u_new.copy()
        if save_all_steps:
            u_hist.append(u.copy())
    
    if save_all_steps:
        return np.array(u_hist)
    else:
        return u

def crank_nicolson_variable_k(u0, k, dx, dt, Nt, save_all_steps=True):
    Nx = len(u0)
    u = u0.copy()
    if save_all_steps:
        u_hist = [u.copy()]
    for n in range(Nt):
        a = np.zeros(Nx - 1)
        b = np.zeros(Nx)
        c = np.zeros(Nx - 1)
        for i in range(1, Nx - 1):
            k_half_right = 0.5 * (k[i] + k[i + 1])
            k_half_left = 0.5 * (k[i] + k[i - 1])
            a[i - 1] = -0.5 * dt / dx**2 * k_half_left
            c[i] = -0.5 * dt / dx**2 * k_half_right
            b[i] = 1 + 0.5 * dt / dx**2 * (k_half_left + k_half_right)
        b[0] = 1.0
        b[-1] = 1.0

        rhs = u.copy()
        for i in range(1, Nx - 1):
            k_half_right = 0.5 * (k[i] + k[i + 1])
            k_half_left = 0.5 * (k[i] + k[i - 1])
            rhs[i] += 0.5 * dt / dx**2 * (
                k_half_right * (u[i + 1] - u[i]) - k_half_left * (u[i] - u[i - 1])
            )

        rhs[0] = 1.0
        rhs[-1] = rhs[-2]

        c_star = np.zeros(Nx - 1)
        d_star = np.zeros(Nx)
        c_star[0] = c[0] / b[0]
        d_star[0] = rhs[0] / b[0]
        for i in range(1, Nx - 1):
            denom = b[i] - a[i - 1] * c_star[i - 1]
            c_star[i] = c[i] / denom
            d_star[i] = (rhs[i] - a[i - 1] * d_star[i - 1]) / denom
        d_star[-1] = (rhs[-1] - a[-2] * d_star[-2]) / (b[-1] - a[-2] * c_star[-2])
        
        u_new = u.copy()
        u_new[-1] = d_star[-1]
        for i in range(Nx - 2, -1, -1):
            u_new[i] = d_star[i] - c_star[i - 1] * u_new[i + 1] if i > 0 else 1.0
        u = u_new.copy()
        if save_all_steps:
            u_hist.append(u.copy())
    
    if save_all_steps:
        return np.array(u_hist)
    else:
        return u

start = time.time()
u_ftcs_var = ftcs_variable_k(u0, k, dx, dt, Nt)

start = time.time()
u_cn_var = crank_nicolson_variable_k(u0, k, dx, dt, Nt)

k_constant = k0 * np.ones_like(k)
u_ftcs_const = ftcs_variable_k(u0, k_constant, dx, dt, Nt)

u_cn_const = crank_nicolson_variable_k(u0, k_constant, dx, dt, Nt)

t_analytical = 0.1
u_analytical = analytical_solution_constant_k(x, t_analytical, k0, L)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
times_to_plot = [0.01, 0.1, 0.5]
for ts in times_to_plot:
    index_ftcs = int(ts / t_total * (len(u_ftcs_var) - 1))
    index_cn = int(ts / t_total * (len(u_cn_var) - 1))
    plt.plot(x, u_ftcs_var[index_ftcs], '--', lw=1.5, label=f'FTCS t={ts:.2f}s')
    plt.plot(x, u_cn_var[index_cn], '-', lw=2, label=f'CN t={ts:.2f}s')

plt.xlabel('x, м')
plt.ylabel('Температура u(x,t)')
plt.title('Переменная теплопроводность: Сравнение методов')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
index_analytical = int(t_analytical / t_total * (len(u_ftcs_const) - 1))
plt.plot(x, u_analytical, 'k-', lw=3, label='Аналитическое')
plt.plot(x, u_ftcs_const[index_analytical], 'r--', lw=2, label='FTCS')
plt.plot(x, u_cn_const[index_analytical], 'b:', lw=2, label='CN')
plt.xlabel('x, м')
plt.ylabel('Температура u(x,t)')
plt.title(f'Постоянная k: Сравнение с аналитическим (t={t_analytical}s)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
error_ftcs = np.abs(u_ftcs_const[index_analytical] - u_analytical)
error_cn = np.abs(u_cn_const[index_analytical] - u_analytical)
plt.semilogy(x, error_ftcs, 'r--', label='Ошибка FTCS')
plt.semilogy(x, error_cn, 'b:', label='Ошибка CN')
plt.xlabel('x, м')
plt.ylabel('Абсолютная ошибка')
plt.title('Ошибки численных методов')
plt.legend()
plt.grid(True, alpha=0.3)

Nx_list = [21, 41, 81, 161, 321]
errors_ftcs = []
errors_cn = []
dx_list = []

CFL_target = 0.4
t_final = 0.1

for Nx in Nx_list:
    dx = L / (Nx - 1)
    dx_list.append(dx)
    
    dt_adaptive = CFL_target * dx**2 / k0
    Nt_adaptive = max(1, int(t_final / dt_adaptive))
    
    x_local = np.linspace(0, L, Nx)
    k_local = k0 * np.ones_like(x_local)
    u0_local = np.zeros(Nx)
    u0_local[0] = 1.0
    
    u_analytical_local = analytical_solution_constant_k(x_local, t_final, k0, L)
    
    u_ftcs_local = ftcs_variable_k(u0_local, k_local, dx, dt_adaptive, Nt_adaptive, save_all_steps=False)
    u_cn_local = crank_nicolson_variable_k(u0_local, k_local, dx, dt_adaptive, Nt_adaptive, save_all_steps=False)
    
    error_ftcs = np.linalg.norm(u_ftcs_local - u_analytical_local) / np.sqrt(Nx)
    error_cn = np.linalg.norm(u_cn_local - u_analytical_local) / np.sqrt(Nx)
    
    errors_ftcs.append(error_ftcs)
    errors_cn.append(error_cn)
    

plt.subplot(2, 2, 4)
dx_array = np.array(dx_list)
errors_ftcs_array = np.array(errors_ftcs)
errors_cn_array = np.array(errors_cn)

log_dx = np.log(dx_array)
log_err_ftcs = np.log(errors_ftcs_array)
log_err_cn = np.log(errors_cn_array)

p_ftcs = np.polyfit(log_dx, log_err_ftcs, 1)[0]
p_cn = np.polyfit(log_dx, log_err_cn, 1)[0]

plt.loglog(dx_array, errors_ftcs_array, 'ro-', label=f'FTCS (p={p_ftcs:.2f})')
plt.loglog(dx_array, errors_cn_array, 'bs-', label=f'CN (p={p_cn:.2f})')
plt.loglog(dx_array, dx_array**1, 'k--', label='O(Δx)')
plt.loglog(dx_array, dx_array**2, 'k:', label='O(Δx²)')

plt.xlabel('Δx')
plt.ylabel('Норма ошибки')
plt.title('Порядок сходимости методов')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

def total_energy(u, dx):
    return np.trapz(u, dx=dx)

energy_ftcs = [total_energy(u_ftcs_var[i], dx) for i in range(len(u_ftcs_var))]
energy_cn = [total_energy(u_cn_var[i], dx) for i in range(len(u_cn_var))]

t_axis_ftcs = np.linspace(0, t_total, len(u_ftcs_var))
t_axis_cn = np.linspace(0, t_total, len(u_cn_var))

plt.figure(figsize=(9, 5))
plt.plot(t_axis_ftcs, energy_ftcs, '--', label='FTCS')
plt.plot(t_axis_cn, energy_cn, '-', label='Crank–Nicolson')
plt.xlabel('Время, с')
plt.ylabel('Полная энергия системы')
plt.title('Изменение тепловой энергии во времени')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()