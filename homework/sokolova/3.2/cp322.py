import numpy as np
import matplotlib.pyplot as plt

def beam_equations_linear(state, x, EJ, q, L):
    w, theta, M, V = state
    
    dw_dx = theta
    dtheta_dx = M / EJ
    dM_dx = V
    dV_dx = -q 
    
    return np.array([dw_dx, dtheta_dx, dM_dx, dV_dx])

def beam_equations_nonlinear(state, x, EJ, q, L):
    w, theta, M, V = state
    
    dw_dx = theta
    dtheta_dx = M / (EJ * (1 + theta**2)**(3/2))
    dM_dx = V * np.cos(theta) - q * w * np.sin(theta)
    dV_dx = -q * np.cos(theta)
    
    return np.array([dw_dx, dtheta_dx, dM_dx, dV_dx])

def rk4_step(state, x, h, EJ, q, L, nonlinear=False):
    if nonlinear:
        k1 = h * beam_equations_nonlinear(state, x, EJ, q, L)
        k2 = h * beam_equations_nonlinear(state + 0.5*k1, x + 0.5*h, EJ, q, L)
        k3 = h * beam_equations_nonlinear(state + 0.5*k2, x + 0.5*h, EJ, q, L)
        k4 = h * beam_equations_nonlinear(state + k3, x + h, EJ, q, L)
    else:
        k1 = h * beam_equations_linear(state, x, EJ, q, L)
        k2 = h * beam_equations_linear(state + 0.5*k1, x + 0.5*h, EJ, q, L)
        k3 = h * beam_equations_linear(state + 0.5*k2, x + 0.5*h, EJ, q, L)
        k4 = h * beam_equations_linear(state + k3, x + h, EJ, q, L)
    
    return state + (k1 + 2*k2 + 2*k3 + k4) / 6

def integrate_beam(theta_guess, EJ, q, L, n_steps=100, nonlinear=False):
    initial_state = np.array([0.0, theta_guess, 0.0, q*L/2])
    
    x = np.linspace(0, L, n_steps)
    h = L / (n_steps - 1)
    
    trajectory = np.zeros((n_steps, 4))
    trajectory[0] = initial_state
    
    for i in range(1, n_steps):
        trajectory[i] = rk4_step(trajectory[i-1], x[i-1], h, EJ, q, L, nonlinear)
    
    return x, trajectory

def shooting_regula_falsi(EJ, q, L, max_iter=50, nonlinear=False):
    theta_left = np.radians(-5.0)
    theta_right = np.radians(5.0)
    
    x_left, traj_left = integrate_beam(theta_left, EJ, q, L, nonlinear=nonlinear)
    x_right, traj_right = integrate_beam(theta_right, EJ, q, L, nonlinear=nonlinear)
    
    residual_left = traj_left[-1, 0]
    residual_right = traj_right[-1, 0]
    
    for iteration in range(max_iter):
        if abs(residual_left) < 1e-8:
            return theta_left, traj_left
        if abs(residual_right) < 1e-8:
            return theta_right, traj_right
        
        theta_new = (residual_right * theta_left - residual_left * theta_right) / (residual_right - residual_left)
        
        x_new, traj_new = integrate_beam(theta_new, EJ, q, L, nonlinear=nonlinear)
        residual_new = traj_new[-1, 0]
        
        if residual_new * residual_left > 0:
            theta_left, residual_left = theta_new, residual_new
        else:
            theta_right, residual_right = theta_new, residual_new
        
        if abs(residual_new) < 1e-8:
            return theta_new, traj_new
    
    return theta_new, traj_new

def shooting_newton(EJ, q, L, theta_initial=np.radians(1.0), max_iter=50, nonlinear=False):
    theta = theta_initial
    h = 1e-6
    
    for iteration in range(max_iter):
        x, traj = integrate_beam(theta, EJ, q, L, nonlinear=nonlinear)
        residual = traj[-1, 0]  # w(L)
        
        if abs(residual) < 1e-8:
            return theta, traj
        
        x_p, traj_p = integrate_beam(theta + h, EJ, q, L, nonlinear=nonlinear)
        residual_p = traj_p[-1, 0]
        
        derivative = (residual_p - residual) / h
        
        theta = theta - residual / derivative
    
    return theta, traj

EJ = 1e5 
L = 1.0 

q_linear = 1000

theta_rf_linear, traj_rf_linear = shooting_regula_falsi(EJ, q_linear, L, nonlinear=False)
theta_n_linear, traj_n_linear = shooting_newton(EJ, q_linear, L, nonlinear=False)

q_nonlinear = 50000 

theta_rf_nonlinear, traj_rf_nonlinear = shooting_regula_falsi(EJ, q_nonlinear, L, nonlinear=True)
theta_n_nonlinear, traj_n_nonlinear = shooting_newton(EJ, q_nonlinear, L, nonlinear=True)

x = np.linspace(0, L, 100)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
w_rf_linear = traj_rf_linear[:, 0] * 1000 
w_n_linear = traj_n_linear[:, 0] * 1000

plt.plot(x, w_rf_linear, 'b-', linewidth=2, label='Regula falsi')
plt.plot(x, w_n_linear, 'r--', linewidth=2, label='Ньютон')
plt.xlabel('x, м')
plt.ylabel('Прогиб w, мм')
plt.title('Линейный случай (q=1000 Н/м)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 2, 2)
w_rf_nonlinear = traj_rf_nonlinear[:, 0] * 1000 
w_n_nonlinear = traj_n_nonlinear[:, 0] * 1000

plt.plot(x, w_rf_nonlinear, 'b-', linewidth=2, label='Regula falsi')
plt.plot(x, w_n_nonlinear, 'r--', linewidth=2, label='Ньютон')
plt.xlabel('x, м')
plt.ylabel('Прогиб w, мм')
plt.title('Нелинейный случай (q=50000 Н/м)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 2, 3)
w_nonlinear_scaled = w_rf_nonlinear * (q_linear / q_nonlinear)

plt.plot(x, w_rf_linear, 'g-', linewidth=2, label='Линейная теория')
plt.plot(x, w_rf_nonlinear, 'm-', linewidth=2, label='Нелинейная теория')
plt.xlabel('x, м')
plt.ylabel('Прогиб w, мм')
plt.title('Сравнение линейной и нелинейной теорий')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 2, 4)

loads = np.linspace(1000, 80000, 20)
max_deflections_linear = []
max_deflections_nonlinear = []

for q_val in loads:
    _, traj_linear = shooting_regula_falsi(EJ, q_val, L, nonlinear=False)
    max_deflections_linear.append(np.max(np.abs(traj_linear[:, 0])) * 1000)
    
    try:
        _, traj_nonlinear = shooting_regula_falsi(EJ, q_val, L, nonlinear=True)
        max_deflections_nonlinear.append(np.max(np.abs(traj_nonlinear[:, 0])) * 1000)
    except:
        max_deflections_nonlinear.append(np.nan)

plt.plot(max_deflections_linear, loads, 'go-', linewidth=2, label='Линейная теория')
plt.plot(max_deflections_nonlinear, loads, 'mo-', linewidth=2, label='Нелинейная теория')
plt.xlabel('Максимальный прогиб, мм')
plt.ylabel('Нагрузка q, Н/м')
plt.title('Диаграмма нагрузка-прогиб')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

