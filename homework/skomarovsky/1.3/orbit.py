import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Model parameters
GM = 1.0

# Initial conditions: x_0, y_0, vx_0, vy_0
sol_0 = np.array((1.0, 0.0, 0.0, 1.2))

# Time mesh
t_0 = 0.0
t_1 = 60
N = 600
tau = (t_1 - t_0) / N  
t = np.linspace(t_0, t_1, N + 1)

# Initialize analytical orbit
N_an = 400
sol_an = np.zeros((N_an + 1, 4), dtype=np.float32)
E = 0.5 * (sol_0[2]**2 + sol_0[3]**2) - GM / (sol_0[0]**2 + sol_0[1]**2)
L = sol_0[0] * sol_0[3] - sol_0[1] * sol_0[2]
a = - GM / (2.0 * E)
e = np.sqrt(1 + 2 * E * L**2 / GM**2)

theta = np.linspace(0, 2*np.pi, N_an + 1)
r = a * (1 - e**2) / (1 + e * np.cos(theta))
sol_an[:,0] = r * np.cos(theta)
sol_an[:,1] = r * np.sin(theta)

# Initialize solutions for Euler and RK4 methods
sol_euler = np.zeros((N + 1, 4), dtype=np.float32)
sol_euler[0] = sol_0

sol_rk4 = np.zeros((N + 1, 4), dtype=np.float32)
sol_rk4[0] = sol_0

# Differential equation:
# x' = v_x
# y' = v_y
# v_x' = -GM * x / (x^2 + y ^2)^(3/2)
# v_y' = -GM * y / (x^2 + y^2)^(3/2)
def f(w):
    x = w[0]
    y = w[1]
    v_x = w[2]
    v_y = w[3]
    r = np.sqrt(x*x + y*y)
    return np.array((v_x, v_y, -GM * x / r**3, -GM * y / r**3))

# Loop to compute solutions
for i in range(0, N):
        sol_euler[i+1] = sol_euler[i] + f(sol_euler[i])*tau
        
        d1 = tau * f(sol_rk4[i])
        d2 = tau * f(sol_rk4[i] + d1/2)
        d3 = tau * f(sol_rk4[i] + d2/2)
        d4 = tau * f(sol_rk4[i] + d3)
        sol_rk4[i+1] = sol_rk4[i] + (d1 + 2*d2 + 2*d3 + d4) / 6.0

# Computing ellipse center and axes for three methods
center_x_an, center_y_an = (np.max(sol_an[:,0]) + np.min(sol_an[:,0]))/2.0, (np.max(sol_an[:,1]) + np.min(sol_an[:,1]))/2.0
a_an, b_an = np.max(sol_an[:,0]) - center_x_an, np.max(sol_an[:,1]) - center_y_an

center_x_euler, center_y_euler = (np.max(sol_euler[:,0]) + np.min(sol_euler[:,0]))/2.0, (np.max(sol_euler[:,1]) + np.min(sol_euler[:,1]))/2.0
a_euler, b_euler = np.max(sol_euler[:,0]) - center_x_euler, np.max(sol_euler[:,1]) - center_y_euler

center_x_rk4, center_y_rk4 = (np.max(sol_rk4[:,0]) + np.min(sol_rk4[:,0]))/2.0, (np.max(sol_rk4[:,1]) + np.min(sol_rk4[:,1]))/2.0
a_rk4, b_rk4 = np.max(sol_rk4[:,0]) - center_x_rk4, np.max(sol_rk4[:,1]) - center_y_rk4

# Error in a, b axes
print("Error in bigger half axis (Euler) |a_an - a_euler| = :", np.abs(a_an - a_euler))
print("Error in bigger half axis (RK4) |a_an - a_rk4| = :", np.abs(a_an - a_rk4))
print("Error in smaller half axis (Euler) |b_an - b_euler| = :", np.abs(b_an - b_euler))
print("Error in smaller half axis (RK4) |b_an - b_rk4| = :", np.abs(b_an - b_rk4))

def show_energy_and_momentum(sol_an, sol_euler, sol_rk4):
        
        fig, axs = plt.subplots(4, layout='constrained')

        # Full Energy : mv^2/2 - GM / r
        energy_euler = 0.5 * (sol_euler[:,2]**2 + sol_euler[:,3]**2) - GM / np.sqrt (sol_euler[:,0]**2 + sol_euler[:,1]**2) 
        energy_rk4 = 0.5 * (sol_rk4[:,2]**2 + sol_rk4[:,3]**2) - GM / np.sqrt (sol_rk4[:,0]**2 + sol_rk4[:,1]**2)

        # Angular Momentum : r x v
        angular_momentum_euler = sol_euler[:,0] * sol_euler[:,3] - sol_euler[:,1] * sol_euler[:,2]
        angular_momentum_rk4 = sol_rk4[:,0] * sol_rk4[:,3] - sol_rk4[:,1] * sol_rk4[:,2]
            
        axs[0].plot(t, energy_euler)
        axs[0].set(xlabel='t', ylabel='E', title='E(t) (Euler)')
        axs[0].set_ylim(1.2 * np.min(energy_euler), 0.8*np.max(energy_euler))
        
        axs[1].plot(t, energy_rk4)
        axs[1].set(xlabel='t', ylabel='E', title='E(t) (RK4)')
        axs[1].set_ylim(1.2 * np.min(energy_rk4), 0.8*np.max(energy_rk4))

        axs[2].plot(t, angular_momentum_euler)
        axs[2].set(xlabel='t', ylabel='L', title='L(t) (Euler)')
        axs[2].set_ylim(0.8*np.min(angular_momentum_euler), 1.2*np.max(angular_momentum_euler))
        
        axs[3].plot(t, angular_momentum_rk4)
        axs[3].set(xlabel='t', ylabel='L', title='L(t) (RK4)')
        axs[3].set_ylim(0.8*np.min(angular_momentum_rk4), 1.2*np.max(angular_momentum_rk4))


def show_planets_move(sol_an, sol_euler, sol_rk4):
    # Body coordinates for three methods
    x_an = sol_an[:,0]
    y_an = sol_an[:,1]

    x_euler = sol_euler[:,0]
    y_euler = sol_euler[:,1]
    traj_euler = (x_euler, y_euler)
    
    x_rk4 = sol_rk4[:,0]
    y_rk4 = sol_rk4[:,1]
    traj_rk4 = (x_rk4, y_rk4)
    
    fig, ax = plt.subplots()

    # Body bobs
    point_euler = ax.plot([], [], marker='o', color='blue',  markersize=8, linestyle='')[0]
    point_rk4 = ax.plot([], [], marker='o', color='red', markersize=8, linestyle='')[0]

    # Body traces
    traj_line_euler = ax.plot([], [], color='blue',  linewidth=2, alpha=0.8, label='Trajectory (Euler)')[0]
    traj_line_rk4 = ax.plot([], [], color='red', linewidth=2, alpha=0.8, label='Trajectory (RK4)')[0]
    traj_line_an = ax.plot(x_an, y_an, 'g--', alpha=0.3, label='Trajectory (Analytical)')[0]

    plt.xlim(1.2 * np.min(x_euler), 1.2 * np.max(x_euler))
    plt.ylim(1.2 * np.min(y_euler), 1.2 * np.max(y_euler))

    ax.set(xlabel='X', ylabel='Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)

    def loop_animation(i):
        # Current coords
        x_cur_euler, y_cur_euler = traj_euler[0][i], traj_euler[1][i]
        x_cur_rk4,   y_cur_rk4   = traj_rk4[0][i],   traj_rk4[1][i]

        # Update body bobs
        point_euler.set_data([x_cur_euler], [y_cur_euler])
        point_rk4.set_data([x_cur_rk4], [y_cur_rk4])

        # Update body traces
        traj_line_euler.set_data(traj_euler[0][:i+1], traj_euler[1][:i+1])
        traj_line_rk4.set_data(traj_rk4[0][:i+1], traj_rk4[1][:i+1])

        return (point_euler, point_rk4, traj_line_euler, traj_line_rk4)

    ax.legend([ 'Point (Euler)', 'Point (RK4)',
         'Trajectory (Euler)', 'Trajectory (RK4)'
        ], loc='best')

    ani = animation.FuncAnimation(
        fig=fig,
        func=loop_animation,
        frames=N,
        interval=20,
        repeat=True,
        repeat_delay=1000
    )
    plt.show()
    
show_energy_and_momentum(sol_an, sol_euler, sol_rk4)
show_planets_move(sol_an, sol_euler, sol_rk4)
