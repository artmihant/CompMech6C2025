import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

angles = [15, 30, 45, 60, 75] 

def calc_traj(alpha):

    # Model parameters
    g = 9.81

    # Initial conditions: x_0, y_0, vx_0, vy_0
    v_0 = 100
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

    return sol_an, sol_euler, sol_rk4


def calc_optimal_angle():
    max_iter = 1000
    iter = 0
    alpha_opt = np.radians(15.0)
    h = np.radians(90.0) / 100

    # Newton method for finding f'(x) = 0:
    # x_n+1 = x_n - f'(x_n)/f''(x_n)
    # derivatives are approximated as 
    # f'(x_n) = f(x_n + h) - f(x_n-h) / 2h
    # f''(x_n) = (f(x_n + h) - 2 * f(x_n) +  f(x_n - h))/ h^2
    while iter < max_iter:
        sol_an_p, sol_euler_p, sol_rk4_p = calc_traj(alpha_opt + h)
        sol_an_m, sol_euler_m, sol_rk4_m = calc_traj(alpha_opt - h)
        sol_an_0, sol_euler_0, sol_rk4_0 = calc_traj(alpha_opt)
    
        f_p = np.max(sol_rk4_p[:,0])
        f_m = np.max(sol_rk4_m[:,0])
        f_0 = np.max(sol_rk4_0[:,0])

        d1 = (f_p - f_m)/ (2.0 * h)
        d2 = (f_p - 2.0 *f_0 + f_m) / (h*h)
        alpha_opt = alpha_opt - d1/d2
        if(np.abs(d1) < 1e-8):
            break
        iter = iter + 1

    alpha_an = np.pi / 4.0

    print('Optimal angle = ', alpha_opt)
    print('Analyticl optimal angle = ', alpha_an)
    print('Delta = ', (alpha_an-alpha_opt)/alpha_an*100, '%')

def show_energy(sol_an, sol_euler, sol_rk4):
        
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

      
    
def animate_multiple_angles(angles_deg):
    cmap = plt.get_cmap('tab10')

    sols = []
    max_len = 0
    max_x = 0.0
    max_y = 0.0

    # Calculating and preserving for all angles
    for k, ang_deg in enumerate(angles_deg):
        alpha = np.radians(ang_deg)
        sol_an, _, sol_rk4 = calc_traj(alpha)

        # Сохраняем
        color = cmap(k % 10)
        sols.append({
            'alpha_deg': ang_deg,
            'an': sol_an,
            'rk4': sol_rk4,
            'color': color,
            'label': f'alpha={ang_deg}'
        })
        max_len = max(max_len, sol_rk4.shape[0])
        max_x = max(max_x, float(np.max(sol_rk4[:,0])))
        max_y = max(max_y, float(np.max(sol_rk4[:,1])))

    fig, ax = plt.subplots()
    ax.set(xlabel='X', ylabel='Y', title='Trajectory (RK4)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.4)

    ax.set_xlim(0, 1.1 * max_x)
    ax.set_ylim(0, 1.1 * max_y)

    trace_lines = [] 
    points = []

    # Making analytical trajectory and initializing trace and body bob for all angles
    for item in sols:
        sol_an = item['an']
        sol_rk4 = item['rk4']
        color = item['color']
        label = item['label']

        ax.plot(sol_an[:,0], sol_an[:,1], linestyle='--', color=color, alpha=0.6, label=f'Analytic {label}')

        # Trace line
        line, = ax.plot([], [], color=color, linewidth=2.0, label=f'RK4 {label}')
        trace_lines.append(line)

        # Body bob
        point, = ax.plot([], [], marker='o', color=color, markersize=7, linestyle='')
        points.append(point)

    ax.legend(ncols=2, fontsize=9)

    def clamp_index(i, n):
        if i < 0:
            return 0
        if i >= n:
            return n - 1
        return i

    def update(i):
        artists = []
        for idx, item in enumerate(sols):
            sol_rk4 = item['rk4']
            n = sol_rk4.shape[0]
            j = clamp_index(i, n)

            x = sol_rk4[:j+1, 0]
            y = sol_rk4[:j+1, 1]

            trace_lines[idx].set_data(x, y)
            points[idx].set_data([x[-1]], [y[-1]])

            artists.extend([trace_lines[idx], points[idx]])

        return artists

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=max_len,
        interval=40,
        blit=True
    )
    plt.show()
    return ani


calc_optimal_angle()
animate_multiple_angles(angles)






 

