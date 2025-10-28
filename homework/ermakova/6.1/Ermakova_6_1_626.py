import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

L = 1.0
k0 = 1.0
T_max = 0.5
times = [0.01, 0.1, 0.5]
cfl = 0.45

def k_var(x):
    return k0 * (1.0 + 0.5 * np.sin(2.0 * np.pi * x / L))

def k_const(x):
    return k0 * np.ones_like(x)

def ftcs_solve(Nx, T_final, kfun, dt_fixed=None):
    dx = L / (Nx - 1)
    x = np.linspace(0.0, L, Nx)
    kx = kfun(x)
    k_half = 0.5 * (kx[1:] + kx[:-1])
    if dt_fixed is None:
        dt = cfl * dx * dx / (2.0 * np.max(k_half))
    else:
        dt = float(dt_fixed)
    Nt = max(1, int(np.ceil(T_final / dt)))
    dt = T_final / Nt
    u = np.zeros(Nx)
    u[0] = 1.0
    for _ in range(Nt):
        F = k_half * (u[1:] - u[:-1]) / dx
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + (dt / dx) * (F[1:] - F[:-1])
        u_new[0] = 1.0
        u_new[-1] = u_new[-2]
        u = u_new
    return x, u

def crank_nicolson_solve(Nx, T_final, kfun, dt_fixed=None):
    dx = L / (Nx - 1)
    x = np.linspace(0.0, L, Nx)
    kx = kfun(x)
    k_half = 0.5 * (kx[1:] + kx[:-1])
    if dt_fixed is None:
        dt = cfl * dx * dx / (2.0 * np.max(k_half))
    else:
        dt = float(dt_fixed)
    Nt = max(1, int(np.ceil(T_final / dt)))
    dt = T_final / Nt
    alpha = dt / (2.0 * dx * dx)
    main = np.ones(Nx)
    lower = np.zeros(Nx - 1)
    upper = np.zeros(Nx - 1)
    main[1:-1] = 1.0 + alpha * (k_half[1:] + k_half[:-1])
    lower[1:] = -alpha * k_half[:-1]
    upper[:-1] = -alpha * k_half[1:]
    main[0] = 1.0
    upper[0] = 0.0
    main[-1] = 1.0
    lower[-1] = -1.0
    A = diags([lower, main, upper], offsets=[-1, 0, 1], format='csc')
    u = np.zeros(Nx)
    u[0] = 1.0
    for _ in range(Nt):
        b = u.copy()
        b[1:-1] = u[1:-1] + alpha * (k_half[1:] * (u[2:] - u[1:-1]) - k_half[:-1] * (u[1:-1] - u[:-2]))
        b[0] = 1.0
        b[-1] = 0.0
        u = spsolve(A, b)
    return x, u

def analytic_u(x, t, terms=600):
    n = np.arange(terms, dtype=float)
    mu = np.pi * (n + 0.5) / L
    w = (-2.0 / (L * mu)) * np.exp(-k0 * (mu ** 2) * t)
    S = np.sin(np.outer(mu, x))
    return 1.0 + w @ S

def energy(u, dx):
    return np.trapz(u, dx=dx)

Nx_plot = 201
plt.figure(figsize=(12, 4))
for i, t in enumerate(times, 1):
    plt.subplot(1, 3, i)
    x_ftcs_v, u_ftcs_v = ftcs_solve(Nx_plot, t, k_var)
    x_cn_v, u_cn_v = crank_nicolson_solve(Nx_plot, t, k_var)
    x = np.linspace(0.0, L, Nx_plot)
    u_an = analytic_u(x, t, terms=800)
    plt.plot(x_ftcs_v, u_ftcs_v, '--', label='FTCS, k(x)')
    plt.plot(x_cn_v, u_cn_v, '-.', label='CN, k(x)')
    plt.plot(x, u_an, '-', label='Аналит., k=k₀')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f't={t}')
    plt.grid(True, alpha=0.3)
    if i == 1:
        plt.legend()
plt.tight_layout()
plt.show()

time_points = np.linspace(1e-4, T_max, 30)
Nx_energy = 151
E_ftcs = []
E_cn = []
for t in time_points:
    _, u_ftcs = ftcs_solve(Nx_energy, t, k_var)
    _, u_cn = crank_nicolson_solve(Nx_energy, t, k_var)
    dx_e = L / (Nx_energy - 1)
    E_ftcs.append(energy(u_ftcs, dx_e))
    E_cn.append(energy(u_cn, dx_e))
plt.figure(figsize=(8, 5))
plt.plot(time_points, E_ftcs, 'o-', markersize=4, label='FTCS, k(x)')
plt.plot(time_points, E_cn, 's-', markersize=4, label='CN, k(x)')
plt.xlabel('t')
plt.ylabel('Энергия')
plt.title('Изменение тепловой энергии')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

Nx_values = [25, 50, 100, 200, 400]
dx_min = L / (Nx_values[-1] - 1)
dt_fixed = cfl * dx_min * dx_min / (2.0 * k0)
err_ftcs = []
err_cn = []
dx_vals = []
T_conv = 0.1
for Nx in Nx_values:
    x, u_ftcs = ftcs_solve(Nx, T_conv, k_const, dt_fixed=dt_fixed)
    _, u_cn = crank_nicolson_solve(Nx, T_conv, k_const, dt_fixed=dt_fixed)
    u_ex = analytic_u(x, T_conv, terms=1200)
    err_ftcs.append(np.sqrt(np.mean((u_ftcs - u_ex) ** 2)))
    err_cn.append(np.sqrt(np.mean((u_cn - u_ex) ** 2)))
    dx_vals.append(L / (Nx - 1))
dx_vals = np.array(dx_vals)
plt.figure(figsize=(8, 5))
plt.loglog(dx_vals, err_ftcs, 'o-', label='FTCS, k=k₀')
plt.loglog(dx_vals, err_cn, 's-', label='CN, k=k₀')
ref = err_ftcs[0] * (dx_vals / dx_vals[0]) ** 2
plt.loglog(dx_vals, ref, '--', label='O(Δx²)')
plt.xlabel('Δx')
plt.ylabel('RMS ошибка')
plt.title('Порядок сходимости по Δx при фиксированном Δt')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.show()
for i in range(1, len(Nx_values)):
    p1 = np.log(err_ftcs[i - 1] / err_ftcs[i]) / np.log(dx_vals[i - 1] / dx_vals[i])
    p2 = np.log(err_cn[i - 1] / err_cn[i]) / np.log(dx_vals[i - 1] / dx_vals[i])
    print(f"FTCS: Nx {Nx_values[i-1]}→{Nx_values[i]}: p ≈ {p1:.3f}")
    print(f"CN:   Nx {Nx_values[i-1]}→{Nx_values[i]}: p ≈ {p2:.3f}")
