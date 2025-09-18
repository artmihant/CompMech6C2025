import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Model parameters
g = 9.81

# Initial conditions: x_0, y_0, vx_0, vy_0
v_0 = 100
alpha = np.radians(30.0)
vx_0  = v_0 * np.cos(alpha)
vy_0 = v_0 * np.sin(alpha)
sol_0 = np.array((0.0, 0.0,vx_0, vy_0), dtype=np.float32)

# Time mesh
t_0 = 0.0
t_1 = 2 * v_0 * np.sin(alpha) / g
N = 30
tau = (t_1 - t_0) / N  
t = np.linspace(t_0, t_1, N + 1)

# Initialize solutions for analytical, Euler and RK4 methods
sol_an = np.zeros((N + 1,4), dtype=np.float32)
x_an = vx_0 * t
y_an = x_an * math.tan(alpha) - g * x_an * x_an / (2.0 * v_0 * v_0 * math.cos(alpha) ** 2)
sol_an[:,0] = x_an
sol_an[:,1] = y_an
sol_an[:,2] = v_0 * np.cos(alpha)
sol_an[:,3] = v_0 * np.sin(alpha) - g * t

sol_euler = np.zeros((N + 1, 4), dtype=np.float32)
sol_euler[0] = sol_0

sol_rk4 = np.zeros((N + 1, 4), dtype=np.float32)
sol_rk4[0] = sol_0

# Differential equation:
# x' = v_x
# y' = v_y
# v_x' = 0
# v_y' = -g
def f(w):
	x = w[0]
	y = w[1]
	v_x = w[2]
	v_y = w[3]
	return np.array((v_x, v_y, 0.0, -g))

# Loop to compute solutions
for i in range(0, N):
        sol_euler[i+1] = sol_euler[i] + f(sol_euler[i])*tau
        
        d1 = tau * f(sol_rk4[i])
        d2 = tau * f(sol_rk4[i] + d1/2)
        d3 = tau * f(sol_rk4[i] + d2/2)
        d4 = tau * f(sol_rk4[i] + d3)
        sol_rk4[i+1] = sol_rk4[i] + (d1 + 2*d2 + 2*d3 + d4)/6.0
	

def show_energy_and_print_error(sol_an, sol_euler, sol_rk4):
        
        fig, axs = plt.subplots(3, layout='constrained')

        # Full Energy : mv^2/2 + mgy
        energy_an = 0.5 * (sol_an[:,2]**2 + sol_an[:,3]**2) + g * sol_an[:,1] 
        energy_euler = 0.5 * (sol_euler[:,2]**2 + sol_euler[:,3]**2) + g * sol_euler[:,1] 
        energy_rk4 = 0.5 * (sol_rk4[:,2]**2 + sol_rk4[:,3]**2) + g * sol_rk4[:,1] 
            
        axs[0].plot(t, energy_an)
        axs[0].set(xlabel='t', ylabel='E', title='E(t) (Analytical)')
        axs[0].set_ylim(0.8 * np.min(energy_an), 1.2*np.max(energy_an))
        
        axs[1].plot(t, energy_euler)
        axs[1].set(xlabel='t', ylabel='E', title='E(t) (Euler)')
        axs[1].set_ylim(0.8 * np.min(energy_an), 1.2*np.max(energy_euler))
        
        axs[2].plot(t, energy_rk4)
        axs[2].set(xlabel='t', ylabel='E', title='E(t) (RK4)')
        axs[2].set_ylim(0.8 * np.min(energy_an), 1.2*np.max(energy_rk4))

        print('MAX(|E_an - E_euler|) = ', np.max(np.abs(energy_an-energy_euler)))
        print('MAX(|E_an - E_rk4|) = ', np.max(np.abs(energy_an-energy_rk4)))
        print('MAX(|x_an - x_euler|) = ', np.max(np.abs(sol_an[:,0]-sol_euler[:,0])))
        print('MAX(|y_an - y_euler|) = ', np.max(np.abs(sol_an[:,1]-sol_euler[:,1])))
        print('MAX(|x_an - x_rk4|) = ', np.max(np.abs(sol_an[:,0]-sol_rk4[:,0])))
        print('MAX(|y_an - y_rk4|) = ', np.max(np.abs(sol_an[:,1]-sol_rk4[:,1])))

        

def show_body_fly(sol_an, sol_euler, sol_rk4):
    # Body coordinates for three methods
    x_an = sol_an[:,0]
    y_an = sol_an[:,1]
    traj_an = (x_an, y_an)

    x_euler = sol_euler[:,0]
    y_euler = sol_euler[:,1]
    traj_euler = (x_euler, y_euler)
    
    x_rk4 = sol_rk4[:,0]
    y_rk4 = sol_rk4[:,1]
    traj_rk4 = (x_rk4, y_rk4)
    
    fig, ax = plt.subplots()

    # Body bobs
    point_an = ax.plot([], [], marker='o', color='red',   markersize=8, linestyle='')[0]
    point_euler = ax.plot([], [], marker='o', color='gold',  markersize=8, linestyle='')[0]
    point_rk4 = ax.plot([], [], marker='o', color='green', markersize=8, linestyle='')[0]

    # Body traces
    traj_line_euler = ax.plot([], [], color='gold',  linewidth=2, alpha=0.8, label='Trajectory (Euler)')[0]
    traj_line_rk4 = ax.plot([], [], color='green', linewidth=2, alpha=0.8, label='Trajectory (RK4)')[0]
    traj_line_an = ax.plot([], [], color='red',   linewidth=2, alpha=0.8, label='Trajectory (Analytical)', linestyle='--')[0]

    plt.xlim(0, 1.2 * np.max([np.max(x_an), np.max(x_euler), np.max(x_rk4)]))
    plt.ylim(0, 1.2 * np.max([np.max(y_an), np.max(y_euler), np.max(y_rk4)]))

    ax.set(xlabel='X', ylabel='Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)

    def loop_animation(i):
        # Current coords
        x_cur_an,    y_cur_an    = traj_an[0][i],    traj_an[1][i]
        x_cur_euler, y_cur_euler = traj_euler[0][i], traj_euler[1][i]
        x_cur_rk4,   y_cur_rk4   = traj_rk4[0][i],   traj_rk4[1][i]

        # Update body bobs
        point_an.set_data([x_cur_an], [y_cur_an])
        point_euler.set_data([x_cur_euler], [y_cur_euler])
        point_rk4.set_data([x_cur_rk4], [y_cur_rk4])

        # Update body traces
        traj_line_an.set_data(traj_an[0][:i+1], traj_an[1][:i+1])
        traj_line_euler.set_data(traj_euler[0][:i+1], traj_euler[1][:i+1])
        traj_line_rk4.set_data(traj_rk4[0][:i+1], traj_rk4[1][:i+1])

        return (point_an, point_euler, point_rk4, traj_line_an, traj_line_euler, traj_line_rk4)

    ax.legend([
        'Point (Analytical)', 'Point (Euler)', 'Point (RK4)',
         'Trajectory (Euler)', 'Trajectory (RK4)','Trajectory (Analytical)'
        ], loc='best')

    ani = animation.FuncAnimation(
        fig=fig,
        func=loop_animation,
        frames=len(x_an),
        interval=50,
        repeat=True,
        repeat_delay=1000
    )
    plt.show()
    
show_energy_and_print_error(sol_an, sol_euler, sol_rk4)
show_body_fly(sol_an, sol_euler, sol_rk4)






 

