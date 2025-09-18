import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

method_name = 'RK45'

# Model parameters
L1 = 0.5
L2 = 1.3
g = 9.81
w1 = np.sqrt(g/L1)
w2 = np.sqrt(g/L2)

# Initial conditions: two angles and two angular velocities
sol_0 = np.array((np.radians(90.0), np.radians(95.0), 0.0, 0.0), dtype=np.float32)

# Time mesh
t_0 = 0.0
t_1 = 30.0
N = 200
tau = (t_1 - t_0) / N  
t = np.linspace(t_0, t_1, N + 1)

# Initialize solution for RK4 method
sol_rk4 = np.zeros((N + 1, 4), dtype=np.float32)
sol_rk4[0] = sol_0

# Differential equation:
# phi1' = psi1
# phi2' = psi2
# psi1' = -2g*phi1/L1 + g*phi2/L1 
# psi2' = -2g*phi2/L2 + 2g*phi1/L2
def f(t,x):
    phi1 = x[0]
    phi2 = x[1]
    psi1 = x[2]
    psi2 = x[3]

    return np.array((psi1, psi2, -2.0*g*phi1/L1 + g*phi2/L1, -2.0*g*phi2/L2 + 2.0*g*phi1/L2))

# Loop to compute three solutions
for i in range(0, N):        
        d1 = tau * f(t[i],sol_rk4[i])
        d2 = tau * f(t[i],sol_rk4[i] + d1/2)
        d3 = tau * f(t[i],sol_rk4[i] + d2/2)
        d4 = tau * f(t[i],sol_rk4[i] + d3)
        sol_rk4[i+1] = sol_rk4[i] + (d1 + 2*d2 + 2*d3 + d4) / 6.0

# Solve for RK45 method
sol_rk45_result = solve_ivp(f, [t_0, t_1], sol_0, method='RK45', t_eval=t, rtol=1e-6, atol=1e-8)
sol_rk45 = sol_rk45_result.y.T.astype(np.float32) 
        
def show_angles(sol):
        phi1 = sol[:,0]
        phi2 = sol[:,1]
        fig, axs = plt.subplots(2, layout='constrained')
           
        axs[0].plot(t, phi1)
        axs[0].set(xlabel='t', ylabel='Phi1', title='Phi1(t)')
        axs[1].plot(t, phi2)
        axs[1].set(xlabel='t', ylabel='Phi2', title='Phi2(t')
        

def show_pendulum_move(sol):
        phi1 = sol[:,0]
        phi2 = sol[:,1]
        
        # Pendulum coordinates
        x1 = L1 * np.sin(phi1)  
        y1 = -L1 * np.cos(phi1)  
        traj1 = (x1, y1)

        x2 = x1 + L2 * np.sin(phi2)  
        y2 = y1 - L2 * np.cos(phi2)  
        traj2 = (x2, y2)
        
        fig, ax = plt.subplots()

        # Bobs and black strings
        line_string1 = ax.plot([], [], 'k-', linewidth=1)[0]
        line_string2 = ax.plot([], [], 'k-', linewidth=1)[0]
        line_trajectory1 = ax.plot([], [], marker='o', color='green', markersize=8)[0]
        line_trajectory2 = ax.plot([], [], marker='o', color='green', markersize=8)[0]

        
        plt.xlim(-1.2 * (L1 + L2), 1.2 * (L1 + L2))
        plt.ylim(-1.2 * (L1 + L2), 1.2 * (L1 + L2))
        
        ax.set(xlabel='X', ylabel='Y', title=f'Pendulum Animation')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)
        ax.plot(0, 0, 'ko', markersize=6)

        def loop_animation(i):
                # Current position of the pendulum bobs
                x1_cur = traj1[0][i]  
                y1_cur = traj1[1][i]

                x2_cur = traj2[0][i]  
                y2_cur = traj2[1][i]
                
                # Pendulum weights (bobs)
                line_trajectory1.set_data([x1_cur], [y1_cur])
                line_trajectory2.set_data([x2_cur], [y2_cur])
                
                # Strings
                line_string1.set_data([0, x1_cur], [0, y1_cur])
                line_string2.set_data([x1_cur, x2_cur], [y1_cur, y2_cur])
                
                return  (line_string1, line_string2, line_trajectory1,  line_trajectory2)

        ax.legend(['String 1', 'String 2', 'Weight 1','Weight 2'])
        ani = animation.FuncAnimation(
                fig=fig,
                func=loop_animation,
                frames=N,
                interval=0.1,
                repeat=True,
                repeat_delay=1000
                )
        plt.show()

if method_name == 'RK4': 
    show_angles(sol_rk4)
    show_pendulum_move(sol_rk4)
if method_name == 'RK45': 
    show_angles(sol_rk45)
    show_pendulum_move(sol_rk45)




 
