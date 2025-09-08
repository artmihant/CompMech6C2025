# Mathematical pendulum modeling using Euler and Runge-Kutta methods
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PHYSICS

gamma = 0
omega = 1

def d(s):
    u = s[0]
    v = s[1]
    return np.array((v, -2*gamma*v-omega**2*np.sin(u)))


u_0 = 0
v_0 = 2

s_0 = np.array((u_0, v_0), dtype=np.float32)

# NUMERIC

delta_t = 0.1

t_min, t_max = 0, 100

steps_number = int((t_max - t_min) / delta_t)

# Make state history array

s_trajectory = np.zeros((steps_number + 1, 2), dtype=np.float32)

s_trajectory[0] = s_0

time_axes = np.linspace(t_min, t_max, steps_number + 1)

# INTEGRATION_METHODS

def explicit_euler_step(s):
    d1 = delta_t * d(s)
    return s + d1

def implicit_euler_step(s):
    d1 = delta_t * d(s)
    d2 = delta_t * d(s + d1)
    return s + d2

def average_exp_imp_step(s):
    # RK2
    d1 = delta_t * d(s)
    d2 = delta_t * d(s + d1)
    return s + (d1 + d2) / 2

def half_point_step(s):
    # RK2
    d1 = delta_t * d(s)
    d2 = delta_t * d(s + d1 / 2)
    return s + d2

def rk4_step(s):
    # RK4
    d1 = delta_t * d(s)
    d2 = delta_t * d(s + d1 / 2)
    d3 = delta_t * d(s + d2 / 2)
    d4 = delta_t * d(s + d3)

    return s + (d1 + 2 * d2 + 2 * d3 + d4) / 6


# SIMULATION
integration_method = rk4_step
method_name = "RK4"

for i in range(1, steps_number + 1):
    s_trajectory[i] = integration_method(s_trajectory[i - 1])


u_trajectory = s_trajectory[:, 0]
v_trajectory = s_trajectory[:, 1]

energy = v_trajectory**2 + omega**2 * 2 * (1 - np.cos(u_trajectory))


# VISUALIZATION

fig, axs = plt.subplots(3, layout='constrained')

axs[0].plot(time_axes, u_trajectory, label='U')
axs[0].plot(time_axes, v_trajectory, label='V')
axs[0].set(xlabel='T', ylabel='U,V', title=f'U(t) и V(t) ({method_name})')

axs[1].plot(time_axes, energy)
axs[1].set(xlabel='T', ylabel='E', title='E(t)')

axs[2].plot(u_trajectory, v_trajectory)
axs[2].set(xlabel='U', ylabel='V', title=f'Фазовая траектория ({method_name})')


def show_pendulum_move(u_trajectory):

    l = 1

    # Coordinates for pendulum: swing from bottom
    x_coords = l * np.sin(u_trajectory)  # Horizontal movement
    y_coords = -l * np.cos(u_trajectory)  # Vertical movement (negative = down)
    trajectory = (x_coords, y_coords)

    fig, ax = plt.subplots()

    # Create empty lines for animation
    line_string = ax.plot([], [], 'k-', linewidth=1)[0]  # Black string
    line_trajectory = ax.plot([], [], marker='o', color='red', markersize=8)[0]  # Red bob

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)

    ax.set(xlabel='X', ylabel='Y', title=f'Анимация маятника ({method_name})')
    ax.set_aspect('equal')  # Make the plot square (equal width and height)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)  # Ground level

    # Mark pivot point
    ax.plot(0, 0, 'ko', markersize=6)  # Black dot for pivot

    def loop_animation(i):
        """ Main computation/animation loop """
        # Current position of the pendulum bob
        x_current = trajectory[0][i]  # Current X coordinate
        y_current = trajectory[1][i]  # Current Y coordinate

        # Draw string from pivot (0,0) to bob (x_current, y_current)
        line_string.set_data([0, x_current], [0, y_current])

        # Draw the bob (pendulum weight)
        line_trajectory.set_data([x_current], [y_current])

        return (line_string, line_trajectory)

    ax.legend(['Нить', 'Грузик маятника'])

    ani = animation.FuncAnimation(
        fig=fig,
        func=loop_animation,
        frames=steps_number,
        interval=0.02,
        repeat=True,
        repeat_delay=1000
    )
    plt.show()

show_pendulum_move(u_trajectory)