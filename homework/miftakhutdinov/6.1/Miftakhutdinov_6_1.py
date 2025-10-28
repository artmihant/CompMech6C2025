import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

plt.style.use('seaborn-v0_8')

L = 1.2
k0 = 0.8
T_final = 0.6
time_points = [0.02, 0.15, 0.4]

def ftcs_solver(N, T_end):
    h = L / (N - 1)
    x = np.linspace(0, L, N)
    k = k0 * (1 + 0.3 * np.cos(3 * np.pi * x))
    
    k_max = np.max(k)
    dt_cfl = 0.4 * h**2 / (2 * k_max)
    Nt = max(1, int(T_end / dt_cfl))
    dt = T_end / Nt
    
    u = np.zeros(N)
    u[0] = 0.8
    k_mid = 0.5 * (k[1:] + k[:-1])
    
    for _ in range(Nt):
        u_next = u.copy()
        for i in range(1, N-1):
            flux_r = k_mid[i] * (u[i+1] - u[i]) / h
            flux_l = k_mid[i-1] * (u[i] - u[i-1]) / h
            u_next[i] = u[i] + dt * (flux_r - flux_l) / h
        u_next[0] = 0.8
        u_next[-1] = u_next[-2]
        u = u_next
    
    return x, u

def cn_solver(N, T_end):
    h = L / (N - 1)
    x = np.linspace(0, L, N)
    k = k0 * (1 + 0.3 * np.cos(3 * np.pi * x))
    dt = 0.002
    Nt = max(1, int(T_end / dt))
    dt = T_end / Nt
    
    u = np.zeros(N)
    u[0] = 0.8
    k_mid = 0.5 * (k[1:] + k[:-1])
    alpha = dt / (2 * h**2)
    
    main = np.ones(N)
    lower = np.zeros(N-1)
    upper = np.zeros(N-1)
    
    main[1:-1] = 1 + alpha * (k_mid[1:] + k_mid[:-1])
    lower[1:] = -alpha * k_mid[:-1]
    upper[:-1] = -alpha * k_mid[1:]
    
    main[0] = 1.0
    upper[0] = 0.0
    main[-1] = 1.0
    lower[-1] = -1.0
    
    A = diags([lower, main, upper], [-1, 0, 1], format='csc')
    
    for _ in range(Nt):
        b = u.copy()
        b[1:-1] += alpha * (k_mid[1:] * (u[2:] - u[1:-1]) - 
                          k_mid[:-1] * (u[1:-1] - u[:-2]))
        b[0] = 0.8
        b[-1] = 0.0
        u = spsolve(A, b)
    
    return x, u

Nx = 120
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, t in enumerate(time_points):
    x_ftcs, u_ftcs = ftcs_solver(Nx, t)
    x_cn, u_cn = cn_solver(Nx, t)
    
    axes[idx].plot(x_ftcs, u_ftcs, color='#FF6B6B', linewidth=2.5, linestyle='-', label='FTCS')
    axes[idx].plot(x_cn, u_cn, color='#4ECDC4', linewidth=2.5, linestyle='--', label='Crank-Nicolson')
    axes[idx].set_xlabel('Coordinate x', fontsize=12)
    axes[idx].set_ylabel('Temperature u(x)', fontsize=12)
    axes[idx].set_title(f'Time = {t}', fontsize=14, fontweight='bold')
    axes[idx].legend(fontsize=11)
    axes[idx].grid(True, alpha=0.4)
    axes[idx].set_facecolor('#F8F9FA')

axes[3].set_visible(False)
plt.tight_layout()
plt.show()

def energy_calc(u, dx):
    return np.trapezoid(u, dx=dx)

t_values = np.linspace(0.005, T_final, 25)
E_ftcs = []
E_cn = []

for t in t_values:
    _, u_ftcs = ftcs_solver(60, t)
    _, u_cn = cn_solver(60, t)
    dx_val = L / 59
    E_ftcs.append(energy_calc(u_ftcs, dx_val))
    E_cn.append(energy_calc(u_cn, dx_val))

plt.figure(figsize=(12, 7))
plt.plot(t_values, E_ftcs, color='#FF6B6B', marker='o', markersize=6, 
         linewidth=2.5, label='FTCS Energy', markevery=3)
plt.plot(t_values, E_cn, color='#4ECDC4', marker='s', markersize=6, 
         linewidth=2.5, label='CN Energy', markevery=3)
plt.xlabel('Time', fontsize=13)
plt.ylabel('System Energy', fontsize=13)
plt.title('Thermal Energy Dynamics', fontsize=15, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.4)
plt.gca().set_facecolor('#F8F9FA')
plt.show()

N_values = [30, 60, 120, 240]
err_ftcs = []
err_cn = []
T_conv = 0.15

N_ref = 480
x_ref, u_ref_ftcs = ftcs_solver(N_ref, T_conv)
_, u_ref_cn = cn_solver(N_ref, T_conv)

for N in N_values:
    x, u_ftcs = ftcs_solver(N, T_conv)
    x, u_cn = cn_solver(N, T_conv)
    u_ftcs_interp = np.interp(x, x_ref, u_ref_ftcs)
    u_cn_interp = np.interp(x, x_ref, u_ref_cn)
    err_ftcs.append(np.sqrt(np.mean((u_ftcs - u_ftcs_interp)**2)))
    err_cn.append(np.sqrt(np.mean((u_cn - u_cn_interp)**2)))

h_values = [L / (N - 1) for N in N_values]

plt.figure(figsize=(12, 8))
plt.loglog(h_values, err_ftcs, color='#FF6B6B', marker='o', markersize=10, 
          linewidth=3, label='FTCS Error')
plt.loglog(h_values, err_cn, color='#4ECDC4', marker='s', markersize=10, 
          linewidth=3, label='CN Error')

for i in range(1, len(N_values)):
    p_ftcs = np.log(err_ftcs[i-1] / err_ftcs[i]) / np.log(h_values[i-1] / h_values[i])
    p_cn = np.log(err_cn[i-1] / err_cn[i]) / np.log(h_values[i-1] / h_values[i])
    print(f"FTCS: N={N_values[i-1]}→{N_values[i]}, order={p_ftcs:.3f}")
    print(f"CN: N={N_values[i-1]}→{N_values[i]}, order={p_cn:.3f}")

ref_line = [err_ftcs[0] * (h/h_values[0])**2 for h in h_values]
plt.loglog(h_values, ref_line, 'k:', linewidth=2.5, label='Theoretical O(h²)', alpha=0.8)

plt.xlabel('Grid Spacing h', fontsize=13)
plt.ylabel('Numerical Error', fontsize=13)
plt.title('Convergence Analysis', fontsize=15, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.4)
plt.gca().set_facecolor('#F8F9FA')
plt.show()

def ftcs_dt_study(N, T_end, factor):
    h = L / (N - 1)
    x = np.linspace(0, L, N)
    k = k0 * (1 + 0.3 * np.cos(3 * np.pi * x))
    
    k_max = np.max(k)
    dt_stable = 0.5 * h**2 / (2 * k_max)
    dt = factor * dt_stable
    Nt = max(1, int(T_end / dt))
    dt = T_end / Nt
    
    u = np.zeros(N)
    u[0] = 0.8
    k_mid = 0.5 * (k[1:] + k[:-1])
    
    for _ in range(Nt):
        u_new = u.copy()
        for i in range(1, N-1):
            flux_r = k_mid[i] * (u[i+1] - u[i]) / h
            flux_l = k_mid[i-1] * (u[i] - u[i-1]) / h
            u_new[i] = u[i] + dt * (flux_r - flux_l) / h
        u_new[0] = 0.8
        u_new[-1] = u_new[-2]
        u = u_new
    
    return x, u

factors = [0.2, 0.6, 1.0, 1.4, 1.8]
T_dt_test = 0.15
N_dt = 60

plt.figure(figsize=(12, 8))
colors = ['#FF9E6D', '#FF6B6B', '#C44569', '#786FA6', '#574B90']

for j, factor in enumerate(factors):
    x_dt, u_dt = ftcs_dt_study(N_dt, T_dt_test, factor)
    plt.plot(x_dt, u_dt, color=colors[j], linewidth=2.5, 
             label=f'CFL = {factor:.1f}')

x_cn_ref, u_cn_ref = cn_solver(N_dt, T_dt_test)
plt.plot(x_cn_ref, u_cn_ref, color='#2C3A47', linewidth=3, 
         linestyle='--', label='CN Reference')

plt.xlabel('Position x', fontsize=13)
plt.ylabel('Temperature Distribution', fontsize=13)
plt.title(f'FTCS Stability: Time Step Effects (t = {T_dt_test})', 
          fontsize=15, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4)
plt.gca().set_facecolor('#F8F9FA')
plt.show()