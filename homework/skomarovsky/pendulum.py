import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Model parameters
L = 1.0
g = 9.81
w_0 = np.sqrt(g/L)

# Initial conditions: angle and angular velocity
sol_0 = np.array((np.radians(20.0), 0.0), dtype=np.float32)

# Time mesh
t_0 = 0.0
t_1 = 10.0
N = 1000
tau = (t_1 - t_0) / N  
t = np.linspace(t_0, t_1, N + 1)

# Initialize solutions for analytical, Euler and RK4 methods
sol_an = np.zeros((N + 1, 2), dtype=np.float32)
sol_an[0] = sol_0

sol_euler = np.zeros((N + 1, 2), dtype=np.float32)
sol_euler[0] = sol_0

sol_rk4 = np.zeros((N + 1, 2), dtype=np.float32)
sol_rk4[0] = sol_0

# Differential equation:
# phi' = psi
# psi' = -w0^2*phi
def f(x):
	phi = x[0]
	psi = x[1]
	return np.array((psi, -w_0*w_0*phi))

# Loop to compute three solutions
for i in range(0, N):
        sol_an[i+1] = np.array((sol_0[1]/w_0*np.sin(w_0*t[i+1]) + sol_0[0]*np.cos(w_0*t[i+1]), sol_0[1] * np.cos(w_0*t[i+1]) - sol_0[0]*w_0*np.sin(w_0*t[i+1])))

        sol_euler[i+1] = sol_euler[i] + f(sol_euler[i])*tau
        
        d1 = tau * f(sol_rk4[i])
        d2 = tau * f(sol_rk4[i] + d1/2)
        d3 = tau * f(sol_rk4[i] + d2/2)
        d4 = tau * f(sol_rk4[i] + d3)
        sol_rk4[i+1] = sol_rk4[i] + (d1 + 2*d2 + 2*d3 + d4)/6.0
	

def show_phase_trajectory(sol_an, sol_euler, sol_rk4):

        fig, axs = plt.subplots(3, layout='constrained')

        axs[0].plot(sol_an[:,0], sol_an[:,1])
        axs[0].set(xlabel='Phi', ylabel='Psi', title=f'Phase Trajectory (Anytical)')

        axs[1].plot(sol_euler[:,0], sol_euler[:,1])
        axs[1].set(xlabel='Phi', ylabel='Psi', title=f'Phase Trajectory (Euler)')

        axs[2].plot(sol_rk4[:,0], sol_rk4[:,1])
        axs[2].set(xlabel='Phi', ylabel='Psi', title=f'Phase Trajectory (RK4)')
        
def show_energy_and_print_error(sol_an, sol_euler, sol_rk4):
        
        fig, axs = plt.subplots(3, layout='constrained')
        
        energy_an = (sol_an[:,1] * L) **2 / 2.0 + g * L *sol_an[:,0]**2 /2.0
        energy_euler = (sol_euler[:,1] * L)**2 / 2.0 + g * L * sol_euler[:,0]**2/2.0
        energy_rk4 = (sol_rk4[:,1] * L)**2 / 2.0 + g * L * sol_rk4[:,0]**2/2.0
            
        axs[0].plot(t, energy_an)
        axs[0].set(xlabel='t', ylabel='E', title='E(t) (Analytical)')
        axs[0].set_ylim(0, 1.2*np.max(energy_an))
        
        axs[1].plot(t, energy_euler)
        axs[1].set(xlabel='t', ylabel='E', title='E(t) (Euler)')
        axs[1].set_ylim(0, 1.2*np.max(energy_euler))
        
        axs[2].plot(t, energy_rk4)
        axs[2].set(xlabel='t', ylabel='E', title='E(t) (RK4)')
        axs[2].set_ylim(0, 1.2*np.max(energy_rk4))

        print('MAX(|E_an - E_euler|) = ', np.max(np.abs(energy_an-energy_euler)))
        print('MAX(|E_an - E_rk4|) = ', np.max(np.abs(energy_an-energy_rk4)))
        print('MAX(|phi_an - phi_euler|) = ', np.max(np.abs(sol_an[:,0]-sol_euler[:,0])))
        print('MAX(|phi_an - phi_rk4|) = ', np.max(np.abs(sol_an[:,0]-sol_rk4[:,0])))


        

def show_pendulum_move(sol_an, sol_euler, sol_rk4):
        phi_an = sol_an[:,0]
        phi_euler = sol_euler[:,0]
        phi_rk4 = sol_rk4[:,0]
        
        # Pendulum coordinates for three methods
        x_an = L * np.sin(phi_an)  
        y_an = -L * np.cos(phi_an)  
        traj_an = (x_an, y_an)
        
        x_euler = L * np.sin(phi_euler)  
        y_euler = -L * np.cos(phi_euler)  
        traj_euler = (x_euler, y_euler)
        

        x_rk4 = L * np.sin(phi_rk4)  
        y_rk4 = -L * np.cos(phi_rk4)  
        traj_rk4 = (x_rk4, y_rk4)
        
        fig, ax = plt.subplots()

        # Bobs and black strings for three methods
        line_string_an = ax.plot([], [], 'k-', linewidth=1)[0] 
        line_trajectory_an = ax.plot([], [], marker='o', color='red', markersize=8)[0]
        line_string_euler = ax.plot([], [], 'k-', linewidth=1)[0] 
        line_trajectory_euler = ax.plot([], [], marker='o', color='yellow', markersize=8)[0]
        line_string_rk4 = ax.plot([], [], 'k-', linewidth=1)[0]
        line_trajectory_rk4 = ax.plot([], [], marker='o', color='green', markersize=8)[0]

        
        plt.xlim(-1.2 * L, 1.2 * L)
        plt.ylim(-1.2 * L, 1.2 * L)
        
        ax.set(xlabel='X', ylabel='Y', title=f'Pendulum Animation')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)
        ax.plot(0, 0, 'ko', markersize=6)

        def loop_animation(i):
                # Current position of the pendulum bobs for three methods
                x_cur_an = traj_an[0][i]
                y_cur_an = traj_an[1][i]  

                x_cur_euler = traj_euler[0][i]  
                y_cur_euler = traj_euler[1][i]  

                x_cur_rk4 = traj_rk4[0][i]  
                y_cur_rk4 = traj_rk4[1][i]
                
                # Pendulum weight (bob)
                line_trajectory_an.set_data([x_cur_an], [y_cur_an])
                line_trajectory_euler.set_data([x_cur_euler], [y_cur_euler])
                line_trajectory_rk4.set_data([x_cur_rk4], [y_cur_rk4])
                
                # String, connecting pivot and bob
                line_string_an.set_data([0, x_cur_an], [0, y_cur_an])
                line_string_euler.set_data([0, x_cur_euler], [0, y_cur_euler])
                line_string_rk4.set_data([0, x_cur_rk4], [0, y_cur_rk4])
                
                return (line_string_an, line_trajectory_an, line_string_euler, line_trajectory_euler, line_string_rk4, line_trajectory_rk4)

        ax.legend(['String (Analytical)', 'Weight (Analytical)','String (Euler)','Weight (Euler)','String RK4','Weight RK4'])
        ani = animation.FuncAnimation(
                fig=fig,
                func=loop_animation,
                frames=N,
                interval=0.02,
                repeat=True,
                repeat_delay=1000
                )
        plt.show()
    
show_phase_trajectory(sol_an, sol_euler, sol_rk4)
show_energy_and_print_error(sol_an, sol_euler, sol_rk4)
show_pendulum_move(sol_an, sol_euler, sol_rk4)




 
