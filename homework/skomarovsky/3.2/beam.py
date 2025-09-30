import numpy as np
import matplotlib.pyplot as plt

# Solve Cauchy problem
def calc_displacements(w_0, theta_0, EJ, L):
    
    # Initial conditions
    sol_0 = np.array((w_0, theta_0), dtype=np.float64)
    
    # Space mesh
    x_0 = 0.0
    x_1 = L
    N = 100
    h = (x_1 - x_0) / N  
    x = np.linspace(x_0, x_1, N + 1)

    # Initialize solution for RK4 method
    sol_rk4 = np.zeros((N + 1, 2), dtype=np.float64)
    sol_rk4[0] = sol_0

    # Uniform force along beam
    q = 5000

    # Moment
    def M(x):
        return q * (L*x - x*x) * 0.5

    # Differential equation:
    # w' = theta
    # theta' = M(x) / EJ
    def f(s, x):
        w = s[0]
        theta = s[1]
        M_x = M(x)
        return np.array((theta, M_x / EJ))

    for i in range(0, N):
        x_i = x[i]
        d1 = h * f(sol_rk4[i], x_i)
        d2 = h * f(sol_rk4[i] + d1/2.0, x_i)
        d3 = h * f(sol_rk4[i] + d2/2.0, x_i)
        d4 = h * f(sol_rk4[i] + d3, x_i)
        sol_rk4[i+1] = sol_rk4[i] + (d1 + 2.0*d2 + 2.0*d3 + d4) / 6.0

    return x, sol_rk4

def shooting_method():
    EJ = 9.82e5
    L = 4.0

    w_0 = 0.0
    w_L = 0.0

    max_iter = 100
    iter_count = 0
    theta = np.radians(1.0)
    h = np.radians(90.0) / 100

    iterations_data = []
    x_mesh = None
    
    print("Shooting method iterations:")
    
    while iter_count < max_iter:
        x, sol_rk4_p = calc_displacements(w_0, theta + h, EJ, L)
        x, sol_rk4_m = calc_displacements(w_0, theta - h, EJ, L)
        x, sol_rk4_0 = calc_displacements(w_0, theta, EJ, L)
        
        if x_mesh is None:
            x_mesh = x
    
        print("iter = ", iter_count, " theta = ", theta)

        f_p = sol_rk4_p[-1,0] - w_L
        f_m = sol_rk4_m[-1,0] - w_L
        f_0 = sol_rk4_0[-1,0] - w_L

        derivative = (f_p - f_m) / (2.0 * h)
        theta = theta - f_0 / derivative
        
        iterations_data.append({
            'iteration': iter_count,
            'theta': theta,
            'trajectory': sol_rk4_0,
            'error': f_0
        })
        
        
        if abs(f_0/derivative) < 1e-8:
            break
        iter_count += 1

    print(f"Final theta: {theta:.6f} rad")
    
    return iterations_data, x_mesh

iterations_data, x = shooting_method()

plt.figure(figsize=(10, 6))

for i, data in enumerate(iterations_data):
    w = data['trajectory'][:, 0] * 1000

    if i == len(iterations_data) - 1:
        plt.plot(x, w, 'b-', linewidth=2, label='final iter')
    else:
        plt.plot(x, w, 'r--', alpha=0.5, label='iter = '+str(i))

plt.xlabel('x')
plt.ylabel('w (x)')
plt.title('Beam Displacemtns')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
