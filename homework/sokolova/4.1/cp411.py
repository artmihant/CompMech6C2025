import numpy as np
import matplotlib.pyplot as plt

sigma = 10
rho = 28
beta = 8/3

def lorenz_system(state):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

def rk4_step(func, state, dt):
    k1 = func(state)
    k2 = func(state + 0.5 * dt * k1)
    k3 = func(state + 0.5 * dt * k2)
    k4 = func(state + dt * k3)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def solve_lorenz(initial_state, t_span, n_points):
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_points
    t_values = np.linspace(t_start, t_end, n_points)
    states = np.zeros((n_points, 3))
    states[0] = initial_state
    
    for i in range(1, n_points):
        states[i] = rk4_step(lorenz_system, states[i-1], dt)
    
    return t_values, states

t_span = (0, 50)
n_points = 10000
initial_state = np.array([1.0, 1.0, 1.0])

t_values, states = solve_lorenz(initial_state, t_span, n_points)
x, y, z = states.T

derivatives = np.array([lorenz_system(state) for state in states])
dx_dt, dy_dt, dz_dt = derivatives.T

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].plot(x, y, 'purple', alpha=0.7, linewidth=0.5)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].set_title('Фазовый портрет (x, y)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, z, 'brown', alpha=0.7, linewidth=0.5)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('z')
axes[0, 1].set_title('Фазовый портрет (x, z)')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(y, z, 'teal', alpha=0.7, linewidth=0.5)
axes[0, 2].set_xlabel('y')
axes[0, 2].set_ylabel('z')
axes[0, 2].set_title('Фазовый портрет (y, z)')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].plot(t_values, x, 'b-', alpha=0.7, linewidth=1)
axes[1, 0].set_xlabel('Время, t')
axes[1, 0].set_ylabel('x(t)')
axes[1, 0].set_title('Координата x(t)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t_values, y, 'r-', alpha=0.7, linewidth=1)
axes[1, 1].set_xlabel('Время, t')
axes[1, 1].set_ylabel('y(t)')
axes[1, 1].set_title('Координата y(t)')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(t_values, z, 'g-', alpha=0.7, linewidth=1)
axes[1, 2].set_xlabel('Время, t')
axes[1, 2].set_ylabel('z(t)')
axes[1, 2].set_title('Координата z(t)')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()