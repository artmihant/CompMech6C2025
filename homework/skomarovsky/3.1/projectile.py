import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Solve Cauchy problem
def calc_traj(alpha, v_0):
    # Model parameters
    g = 9.81

    # Initial conditions: x_0, y_0, vx_0, vy_0
    vx_0  = v_0 * np.cos(alpha)
    vy_0 = v_0 * np.sin(alpha)
    sol_0 = np.array((0.0, 0.0, vx_0, vy_0), dtype=np.float32)

    # Time mesh
    t_0 = 0.0
    t_1 = 2 * v_0 * np.sin(alpha) / g
    N = 100
    tau = (t_1 - t_0) / N  
    t = np.linspace(t_0, t_1, N + 1)

    # Initialize solution for RK4 method
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
            d1 = tau * f(sol_rk4[i])
            d2 = tau * f(sol_rk4[i] + d1/2)
            d3 = tau * f(sol_rk4[i] + d2/2)
            d4 = tau * f(sol_rk4[i] + d3)
            sol_rk4[i+1] = sol_rk4[i] + (d1 + 2*d2 + 2*d3 + d4)/6.0

    return sol_rk4

def calc_desired_angle():
    # Shooting method: we find such angle alpha that we land on the desired range
    v_0 = 100
    g = 9.81
    analytical_desired_alpha = np.radians(34.2345)
    desired_range = v_0**2/g * np.sin(2.0 * analytical_desired_alpha)

    max_iter = 1000
    iter = 0
    desired_alpha = np.radians(42.0)
    h = np.radians(90.0) / 100

    # Newton method for finding f(x) = 0:
    # x_n+1 = x_n - f(x_n)/f'(x_n)
    # derivative is approximated as 
    # f'(x_n) = f(x_n + h) - f(x_n-h) / 2h
    print('Desired range = ', desired_range)
    print('Start shooting method iterations')
    
    # Store iterations for animation
    iterations_data = []
    
    while iter < max_iter:
        print('Iteration = ', iter,' Current angle (radians) = ', desired_alpha)
        sol_rk4_p = calc_traj(desired_alpha + h, v_0)
        sol_rk4_m = calc_traj(desired_alpha - h, v_0)
        sol_rk4_0 = calc_traj(desired_alpha, v_0)
    
        f_p = np.max(sol_rk4_p[:,0]) - desired_range
        f_m = np.max(sol_rk4_m[:,0]) - desired_range
        f_0 = np.max(sol_rk4_0[:,0]) - desired_range

        d1 = (f_p - f_m)/ (2.0 * h)
        desired_alpha = desired_alpha - f_0/d1
        
        # Store iteration data for animation
        iterations_data.append({
            'iteration': iter,
            'alpha': desired_alpha,
            'trajectory': sol_rk4_0,
            'error': f_0
        })
        
        if(np.abs(f_0/d1) < 1e-4):
            break
        iter = iter + 1

    print('Desired angle (radians) = ', desired_alpha)
    print('Analytical desired angle (radians) = ', analytical_desired_alpha)
    
    return iterations_data, desired_range

def animate_iterations(iterations_data, desired_range):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(xlabel='X', ylabel='Y', title='Shooting Method Iterations')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Calculate bounds for plotting
    max_x = max(np.max(data['trajectory'][:, 0]) for data in iterations_data)
    max_y = max(np.max(data['trajectory'][:, 1]) for data in iterations_data)
    
    ax.set_xlim(0, 1.1 * max_x)
    ax.set_ylim(0, 1.1 * max_y)
    
    # Plot desired range marker
    ax.axvline(x=desired_range, color='red', linestyle=':', alpha=0.7, linewidth=2, label=f'Desired range = {desired_range:.2f}')
    ax.plot(desired_range, 0, marker='o', color='red', markersize=10, linestyle='')
    
    # Color map for different iterations
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(iterations_data)) for i in range(len(iterations_data))]
    
    # Plot all trajectories
    for i, data in enumerate(iterations_data):
        alpha_deg = np.degrees(data['alpha'])
        ax.plot(data['trajectory'][:, 0], data['trajectory'][:, 1], 
                color=colors[i], alpha=0.7, linewidth=1.5,
                label=f'iter={data["iteration"]}, alpha={data["alpha"]:.4f}rad')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

iterations_data, desired_range = calc_desired_angle()
animate_iterations(iterations_data, desired_range)
