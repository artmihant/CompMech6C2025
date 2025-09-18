import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

method_name = 'RK4'

# Model parameters
G = 1.0
m1 = 1.0
m2 = 0.1
m3 = 0.2

# Initial conditions: two angles and two angular velocities
sol_0 = np.array((-0.1, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0, -0.3, 0.0, 0.27, -0.95, 0.0), dtype=np.float32)

# Time mesh
t_0 = 0.0
t_1 = 20.0
N = 2000
tau = (t_1 - t_0) / N  
t = np.linspace(t_0, t_1, N + 1)

# Initialize solution for RK4 method
sol_rk4 = np.zeros((N + 1, 12), dtype=np.float32)
sol_rk4[0] = sol_0

# Differential equation:
# x1' = vx1
# y1' = vy1
# x2' = vx2
# y2' = vy2
# x3' = vx3
# y3' = vy3
# vx1' = G*m2*(x2-x1)/r12^3 + G*m3*(x3-x1)/r13^3
# vy1' = G*m2*(y2-y1)/r12^3 + G*m3*(y3-y1)/r13^3
# vx2' = G*m1*(x1-x2)/r12^3 + G*m3*(x3-x2)/r23^3
# vy2' = G*m1*(y1-y2)/r12^3 + G*m3*(y3-y2)/r23^3
# vx3' = G*m1*(x1-x3)/r13^3 + G*m2*(x2-x3)/r23^3
# vy3' = G*m1*(y1-y3)/r13^3 + G*m2*(y2-y3)/r23^3
def f(t,y):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = y
    
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    
    return np.array([vx1, vy1, vx2, vy2, vx3, vy3,
        G*m2*(x2-x1)/r12**3 + G*m3*(x3-x1)/r13**3,
        G*m2*(y2-y1)/r12**3 + G*m3*(y3-y1)/r13**3,
        G*m1*(x1-x2)/r12**3 + G*m3*(x3-x2)/r23**3,
        G*m1*(y1-y2)/r12**3 + G*m3*(y3-y2)/r23**3,
        G*m1*(x1-x3)/r13**3 + G*m2*(x2-x3)/r23**3,
        G*m1*(y1-y3)/r13**3 + G*m2*(y2-y3)/r23**3
    ])

# Loop to compute three solutions
for i in range(0, N):        
        d1 = tau * f(t[i],sol_rk4[i])
        d2 = tau * f(t[i],sol_rk4[i] + d1/2)
        d3 = tau * f(t[i],sol_rk4[i] + d2/2)
        d4 = tau * f(t[i],sol_rk4[i] + d3)
        sol_rk4[i+1] = sol_rk4[i] + (d1 + 2*d2 + 2*d3 + d4) / 6.0
        
def show_velocities(sol):
        v1 = sol[:,6]**2 + sol[:,7]**2
        v2 = sol[:,8]**2 + sol[:,9]**2
        v3 = sol[:,10]**2 + sol[:,11]**2
        fig, axs = plt.subplots(3, layout='constrained')
           
        axs[0].plot(t, v1)
        axs[0].set(xlabel='t', ylabel='v1', title='||v1||(t)')
        axs[1].plot(t, v2)
        axs[1].set(xlabel='t', ylabel='v2', title='||v2||(t)')
        axs[2].plot(t, v3)
        axs[2].set(xlabel='t', ylabel='v3', title='||v3||(t)')

def show_bodies_move(sol):
        
        # Pendulum coordinates 
        traj1 = (sol[:,0],sol[:,1])
        traj2 = (sol[:,2],sol[:,3])
        traj3 = (sol[:,4],sol[:,5])
        
        fig, ax = plt.subplots()

        # Bobs and black strings
        line_trajectory1 = ax.plot([], [], marker='o', color='green', markersize=8)[0]
        line_trajectory2 = ax.plot([], [], marker='o', color='red', markersize=8)[0]
        line_trajectory3 = ax.plot([], [], marker='o', color='blue', markersize=8)[0]

        # Calculate limits for the plot
        x_min = min(np.min(sol[:,0]), np.min(sol[:,2]), np.min(sol[:,4]))
        x_max = max(np.max(sol[:,0]), np.max(sol[:,2]), np.max(sol[:,4]))
        y_min = min(np.min(sol[:,1]), np.min(sol[:,3]), np.min(sol[:,5]))
        y_max = max(np.max(sol[:,1]), np.max(sol[:,3]), np.max(sol[:,5]))

        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.1 * max(x_range, y_range)
        
        x_lim = (x_min - padding, x_max + padding)
        y_lim = (y_min - padding, y_max + padding)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        ax.set(xlabel='X', ylabel='Y', title=f'Body Movement Animation')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        def loop_animation(i):
                # Current position of the body bobs
                x1_cur = traj1[0][i]  
                y1_cur = traj1[1][i]

                x2_cur = traj2[0][i]  
                y2_cur = traj2[1][i]

                x3_cur = traj3[0][i]  
                y3_cur = traj3[1][i]
                
                # Pendulum weights (bobs)
                line_trajectory1.set_data([x1_cur], [y1_cur])
                line_trajectory2.set_data([x2_cur], [y2_cur])
                line_trajectory3.set_data([x3_cur], [y3_cur])
                
                
                return  (line_trajectory1, line_trajectory2, line_trajectory3)

        ax.legend(['Body 1','Body 2', 'Body 3'])
        ani = animation.FuncAnimation(
                fig=fig,
                func=loop_animation,
                frames=N,
                interval=0.02,
                repeat=True,
                repeat_delay=1000
                )
        plt.show()

show_velocities(sol_rk4)
show_bodies_move(sol_rk4)

