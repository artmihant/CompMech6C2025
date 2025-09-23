import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

method_name = 'RK45'

# Model parameters
L1 = 1.3
L2 = 0.5
g = 9.81
w1 = np.sqrt(g/L1)
w2 = np.sqrt(g/L2)

# Base initial conditions
base_sol_0 = np.array((np.radians(80.0), np.radians(-20.0), 0.0, 0.0), dtype=np.float32)

# Create multiple pendulums with small variations in angular velocities
velocity_variations = [-0.1, -0.01, 0.0, 0.01, 0.1]
initial_conditions = []

for d_psi in velocity_variations:
    modified_ic = base_sol_0.copy()
    modified_ic[2] += d_psi
    modified_ic[3] += d_psi  
    initial_conditions.append(modified_ic)

# Time mesh
t_0 = 0.0
t_1 = 100.0
N = 2000
tau = (t_1 - t_0) / N  
t = np.linspace(t_0, t_1, N + 1)

# Differential equation for nonlinear case:
def f(t, x):
    phi1 = x[0]
    phi2 = x[1]
    psi1 = x[2]
    psi2 = x[3]

    delta_phi = phi1 - phi2
    cos_delta = np.cos(delta_phi)
    sin_delta = np.sin(delta_phi)
    sin_phi1 = np.sin(phi1)
    sin_phi2 = np.sin(phi2)
    
    det = L1**2 * L2**2 * (1 + sin_delta**2)

    b1 = -L1 * L2 * psi2**2 * sin_delta - 2 * g * L1 * sin_phi1
    b2 = L1 * L2 * psi1**2 * sin_delta - g * L2 * sin_phi2

    acc1 = (b1 * L2**2 - b2 * L1 * L2 * cos_delta) / det
    acc2 = (b2 * 2 * L1**2 - b1 * L1 * L2 * cos_delta) / det
    
    return np.array((psi1, psi2, acc1, acc2))

# Solve for all pendulums
solutions = []
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

for i, ic in enumerate(initial_conditions):
    
    if method_name == 'RK4':
        sol = np.zeros((N + 1, 4), dtype=np.float32)
        sol[0] = ic
        
        for j in range(0, N):        
            d1 = tau * f(t[j], sol[j])
            d2 = tau * f(t[j], sol[j] + d1/2)
            d3 = tau * f(t[j], sol[j] + d2/2)
            d4 = tau * f(t[j], sol[j] + d3)
            sol[j+1] = sol[j] + (d1 + 2*d2 + 2*d3 + d4) / 6.0
            
        solutions.append(sol)
        
    if method_name == 'RK45':
        sol_result = solve_ivp(f, [t_0, t_1], ic, method='RK45', t_eval=t, rtol=1e-6, atol=1e-8)
        solutions.append(sol_result.y.T.astype(np.float32))

def show_angles(solutions):
    fig, axs = plt.subplots(2, layout='constrained', figsize=(10, 8))
    
    phi1_init = solutions[2][:,0]
    phi2_init = solutions[2][:,1]
    for i, sol in enumerate(solutions):
        phi1 = sol[:,0]
        phi2 = sol[:,1]
        color = colors[i % len(colors)]
        label = f'd_psi={velocity_variations[i]}'
        
        axs[0].plot(t, phi1, color=color, label=label, alpha=0.7)
        axs[1].plot(t, phi2, color=color, label=label, alpha=0.7)

        print('Max deviation of phi1 from initial solution: ', np.max(np.abs(phi1 - phi1_init)) / np.max(phi1_init) * 100,'%')
        print('Max deviation of phi2 from initial solution: ', np.max(np.abs(phi2 - phi2_init))/ np.max(phi2_init) * 100,'%')
    
    axs[0].set(xlabel='t', ylabel='Phi1', title='Phi1(t) for different pendulums')
    axs[1].set(xlabel='t', ylabel='Phi2', title='Phi2(t) for different pendulums')
    axs[0].legend()
    axs[1].legend()

def show_pendulum_move(solutions):
    # Prepare trajectory data for all pendulums
    all_trajectories = []
    
    for sol in solutions:
        phi1 = sol[:,0]
        phi2 = sol[:,1]
        
        # Pendulum coordinates
        x1 = L1 * np.sin(phi1)  
        y1 = -L1 * np.cos(phi1)  
        traj1 = (x1, y1)

        x2 = x1 + L2 * np.sin(phi2)  
        y2 = y1 - L2 * np.cos(phi2)  
        traj2 = (x2, y2)
        
        all_trajectories.append((traj1, traj2))
    
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create artists for all pendulums
    string_lines = []
    bob_lines = []
    
    for i in range(len(solutions)):
        color = colors[i % len(colors)]
        # Strings
        string1, = ax.plot([], [], '-', color=color, linewidth=1, alpha=0.7)
        string2, = ax.plot([], [], '-', color=color, linewidth=1, alpha=0.7)
        # Bobs
        bob1, = ax.plot([], [], 'o', color=color, markersize=6, alpha=0.7)
        bob2, = ax.plot([], [], 'o', color=color, markersize=6, alpha=0.7)
        
        string_lines.append((string1, string2))
        bob_lines.append((bob1, bob2))
    
    plt.xlim(-1.2 * (L1 + L2), 1.2 * (L1 + L2))
    plt.ylim(-1.2 * (L1 + L2), 1.2 * (L1 + L2))
    
    ax.set(xlabel='X', ylabel='Y', title='Multiple Pendulums Animation')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)
    ax.plot(0, 0, 'ko', markersize=8)  # Pivot point

    # Create legend
    legend_elements = []
    for i, d_psi in enumerate(velocity_variations):
        color = colors[i % len(colors)]
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=f'd_psi={d_psi}'))
    ax.legend(handles=legend_elements)

    def loop_animation(i):
        artists = []
        
        for j, ((traj1, traj2), (string1, string2), (bob1, bob2)) in enumerate(zip(all_trajectories, string_lines, bob_lines)):
            # Current position of the pendulum bobs
            x1_cur = traj1[0][i]  
            y1_cur = traj1[1][i]
            x2_cur = traj2[0][i]  
            y2_cur = traj2[1][i]
            
            # Update strings
            string1.set_data([0, x1_cur], [0, y1_cur])
            string2.set_data([x1_cur, x2_cur], [y1_cur, y2_cur])
            
            # Update bobs
            bob1.set_data([x1_cur], [y1_cur])
            bob2.set_data([x2_cur], [y2_cur])
            
            artists.extend([string1, string2, bob1, bob2])
        
        return artists

    ani = animation.FuncAnimation(
        fig=fig,
        func=loop_animation,
        frames=N,
        interval=20,
        repeat=True,
        repeat_delay=1000,
        blit=True
    )
    
    plt.show()
    
    return ani

# Show results
show_angles(solutions)
ani = show_pendulum_move(solutions)