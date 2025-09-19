import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

method_name = 'RK4'

# Model parameters
g = 9.81
frequencies_count = 13
w_first = 6.0
w_last = 12.0
w = np.linspace(w_first, w_last, frequencies_count)

# w = w_first*2**(np.arange(frequencies_count)/(frequencies_count-1))

L = g / w**2

# Initial conditions: angle and angular velocity
sol_0 = np.array((np.radians(90.0), np.radians(0)), dtype=np.float32)

# Time mesh
t_0 = 0
t_1 = 15
N = 240*(t_1-t_0)
tau = (t_1 - t_0) / N
t = np.linspace(t_0, t_1, N + 1)

# Initialize solutions for Euler and RK4 methods
sol_euler = [np.zeros((N + 1, 2), dtype=np.float32) for _ in range(frequencies_count)]
for k in range(frequencies_count):
    sol_euler[k][0] = sol_0

sol_rk4 = [np.zeros((N + 1, 2), dtype=np.float32) for _ in range(frequencies_count)]
for k in range(frequencies_count):
    sol_rk4[k][0] = sol_0

# Differential equation:
# phi' = psi
# psi' = -w0^2*sin(phi)
def f(x, w_0):
    phi = x[0]
    psi = x[1]
    return np.array((psi, -w_0*w_0*np.sin(phi)), dtype=np.float32)

# Loop to compute three solutions
for k in range(frequencies_count):
    w_0 = w[k]
    if method_name == 'Euler':
        for i in range(0, N):
            sol_euler[k][i+1] = sol_euler[k][i] + f(sol_euler[k][i], w_0)*tau
    elif method_name == 'RK4':
        for i in range(0, N):
            d1 = tau * f(sol_rk4[k][i], w_0)
            d2 = tau * f(sol_rk4[k][i] + d1/2, w_0)
            d3 = tau * f(sol_rk4[k][i] + d2/2, w_0)
            d4 = tau * f(sol_rk4[k][i] + d3, w_0)
            sol_rk4[k][i+1] = sol_rk4[k][i] + (d1 + 2*d2 + 2*d3 + d4)/6.0


# Approximating the period of pendulum system by cutting off 
# at the moment t when angle of all pendulums is 0.1 degree away from the initial one
epsilon = np.radians(0.1)
initial_angles = [sol_rk4[k][0, 0] for k in range(frequencies_count)]

cut_index = N-1
for i in range(N//2, N+1):
    all_ok = True
    for k in range(frequencies_count):
        if abs(sol_rk4[k][i, 0] - initial_angles[k]) >= epsilon:
            all_ok = False
            break
    if all_ok:
        cut_index = i
        break

print('Approximate period of all pendulums is: ', t[cut_index+1], cut_index)

def show_phase_trajectory(sols, w_vals):
        fig, axs = plt.subplots(1, layout='constrained')
        colors = plt.cm.plasma(np.linspace(0, 1, len(sols)))
        handles = []
        labels = []
        for idx, sol in enumerate(sols):
            h, = axs.plot(sol[:,0], sol[:,1], color=colors[idx], linewidth=2)
            handles.append(h)
            labels.append(f'w = {w_vals[idx]:.4f}')
            axs.set(xlabel='Phi', ylabel='Psi', title=f'Phase Trajectory')
        axs.legend(handles, labels, loc='best')

def show_coordinate(sols, w_vals):
        fig, axs = plt.subplots(2, layout='constrained')
        colors = plt.cm.plasma(np.linspace(0, 1, len(sols)))
        handles_x = []
        labels_x = []
        for idx, sol in enumerate(sols):
            h, = axs[0].plot(t, L[idx] * np.sin(sol[:,0]), color=colors[idx], linewidth=2)
            handles_x.append(h)
            labels_x.append(f'w = {w_vals[idx]:.2f}')
            axs[0].set(xlabel='t', ylabel='x', title=f'X(t)')
        axs[0].legend(handles_x, labels_x, loc='best')
        handles_y = []
        labels_y = []
        for idx, sol in enumerate(sols):
            h, = axs[1].plot(t, -L[idx] * np.cos(sol[:,1]), color=colors[idx], linewidth=2)
            handles_y.append(h)
            labels_y.append(f'w = {w_vals[idx]:.2f}')
            axs[1].set(xlabel='t', ylabel='y', title=f'Y(t)')
        axs[1].legend(handles_y, labels_y, loc='best')

def show_pendulum_move(sols, w_vals):
    fig, ax = plt.subplots()
    lines_string = []
    lines_point = []

    max_L = np.max(L)
    colors = plt.cm.plasma(np.linspace(0, 1, len(sols)))

    for idx, sol in enumerate(sols):
        phi = sol[:,0]
        x = L[idx] * np.sin(phi)
        y = -L[idx] * np.cos(phi)
        lines_string.append(ax.plot([], [], 'k-', linewidth=1)[0])
        lines_point.append(ax.plot([], [], marker='o', color=colors[idx], markersize=6, linestyle='')[0])

    plt.xlim(-1.2 * max_L, 1.2 * max_L)
    plt.ylim(-1.2 * max_L, 1.2 * max_L)

    ax.set(xlabel='X', ylabel='Y', title=f'Pendulum Animation')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)
    ax.plot(0, 0, 'ko', markersize=6)

    # legend_handles = []
    # legend_labels = []
    # for idx in range(len(sols)):
    #     legend_handles.append(lines_point[idx])
    #     legend_labels.append(f'w = {w_vals[idx]:.2f}')
    # ax.legend(legend_handles, legend_labels, loc='best')

    def loop_animation(i):
        artists = []
        for idx, sol in enumerate(sols):
            phi = sol[:,0]
            x = L[idx] * np.sin(phi)
            y = -L[idx] * np.cos(phi)
            x_cur = x[i]
            y_cur = y[i]
            lines_point[idx].set_data([x_cur], [y_cur])
            lines_string[idx].set_data([0, x_cur], [0, y_cur])
            artists.extend([lines_string[idx], lines_point[idx]])
        return artists

    ani = animation.FuncAnimation(
    fig=fig,
    func=loop_animation,
    frames=range(0, cut_index + 1, 1),
    interval=5,
    blit=True,
    repeat=True,
    repeat_delay=50
    )
    plt.show()

if method_name == 'Euler':
    show_phase_trajectory([sol_euler[k] for k in range(frequencies_count)], w)
    show_coordinate([sol_euler[k] for k in range(frequencies_count)], w)
    show_pendulum_move([sol_euler[k] for k in range(frequencies_count)], w)
elif method_name == 'RK4':
    show_phase_trajectory([sol_rk4[k] for k in range(frequencies_count)], w)
    show_coordinate([sol_rk4[k] for k in range(frequencies_count)], w)
    show_pendulum_move([sol_rk4[k] for k in range(frequencies_count)], w)
