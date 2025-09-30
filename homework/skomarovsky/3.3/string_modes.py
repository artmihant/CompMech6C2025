import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Solve Cauchy problem
def calc_displacements(y1_0, y2_0, lambda_0, L):
    # Initial conditions
    sol_0 = np.array((y1_0, y2_0), dtype=np.float32)

    # Space mesh
    x_0 = 0.0
    x_1 = L
    N = 1000
    h = (x_1 - x_0) / N  
    x = np.linspace(x_0, x_1, N + 1)

    # Initialize solution for RK4 method
    sol_rk4 = np.zeros((N + 1, 2), dtype=np.float32)
    sol_rk4[0] = sol_0

    # Differential equation:
    # y1' = y2
    # y2' = -lambda_0*y1
    def f(s):
        y1 = s[0]
        y2 = s[1]
        return np.array((y2, -lambda_0*y1))

    # Loop to compute solutions
    for i in range(0, N):
        d1 = h * f(sol_rk4[i])
        d2 = h * f(sol_rk4[i] + d1/2)
        d3 = h * f(sol_rk4[i] + d2/2)
        d4 = h * f(sol_rk4[i] + d3)
        sol_rk4[i+1] = sol_rk4[i] + (d1 + 2*d2 + 2*d3 + d4)/6.0

    return sol_rk4

# Shooting method: we find such lambdas that y'' + lambda*y=0, y(0)=0, y(L)=0
# has non-trivial solution (y'(0)!=0, e.g. y'(0)=1)
def calc_desired_angle(lambda_0, L, mode_index):

    y_0 = 0.0
    y_L = 0.0

    max_iter = 1000
    iter = 0
    dy_dx_0 = 1.0
    h = 0.01

    # Newton method for finding f(x) = 0:
    # x_n+1 = x_n - f(x_n)/f'(x_n)
    # derivative is approximated as 
    # f'(x_n) = f(x_n + h) - f(x_n-h) / 2h
    print("=" * 30)
    print('Start shooting method iterations to find mode ', mode_index)

    mode_displacements = []

    # Initial guess
    eigen_value = lambda_0

    while iter < max_iter:
        print('Iteration = ', iter, ' Current lambda = ', eigen_value)
        sol_rk4_p = calc_displacements(y_0, dy_dx_0, eigen_value + h, L)
        sol_rk4_m = calc_displacements(y_0, dy_dx_0, eigen_value - h, L)
        sol_rk4_0 = calc_displacements(y_0, dy_dx_0, eigen_value, L)
    
        f_p = sol_rk4_p[-1, 0] - y_L
        f_m = sol_rk4_m[-1, 0] - y_L
        f_0 = sol_rk4_0[-1, 0] - y_L

        d1 = (f_p - f_m) / (2.0 * h)
        eigen_value = eigen_value - f_0 / d1
        
        if np.abs(f_0) < 1e-4:
            mode_displacements = sol_rk4_0
            break

        iter = iter + 1
    print("=" * 30)
    
    return eigen_value, mode_displacements

# Find several modes
def get_modes(n_modes, lambda_range, L=1.0):    
    y_0 = 0.0
    y_L = 0.0
    dy_dx_0 = 1.0
    
    # Create lambda search grid
    n_points = 20
    lambda_grid = np.linspace(0.1, lambda_range, n_points)
    
    # Calculate boundary residuals for all grid points
    residuals = np.zeros(n_points)
    for i, lambda_val in enumerate(lambda_grid):
        sol = calc_displacements(y_0, dy_dx_0, lambda_val, L)
        residuals[i] = sol[-1, 0] - y_L
    
    # If residual changes sign, then the mode is on the interval
    sign_changes = []
    for i in range(len(residuals) - 1):
        if residuals[i] * residuals[i + 1] < 0:
            sign_changes.append((lambda_grid[i], lambda_grid[i + 1]))
    
    
    # Find eigenvalues using shooting method for each interval
    eigenvalues = []
    eigenmodes = []
    
    for i, (lambda_low, lambda_high) in enumerate(sign_changes):
        if i >= n_modes:
            break
            
        # Use midpoint of interval as initial guess
        lambda_guess = (lambda_low + lambda_high) / 2.0
        print(f"Searching for mode {i} in interval [{lambda_low:.3f}, {lambda_high:.3f}]")
        
        eigenvalue, eigenmode = calc_desired_angle(lambda_guess, L, i)
        eigenvalues.append(eigenvalue)
        eigenmodes.append(eigenmode)
    
    return np.array(eigenvalues), eigenmodes

eigenvalues, eigenmodes = get_modes(n_modes=5, lambda_range=250.0, L=1.0)
    
print("=" * 50)
for i, (eigenval, mode) in enumerate(zip(eigenvalues, eigenmodes)):
    print(f"Mode {i}: eigenvalue = {eigenval:.6f}")
    print(f"  Analytical: {((i+1) * np.pi)**2:.6f}")
    
# Plot the eigenmodes
x = np.linspace(0, 1.0, len(eigenmodes[0]))
plt.figure(figsize=(12, 8))
    
for i, mode in enumerate(eigenmodes):
    plt.plot(x, mode[:, 0], label=f'Mode {i}, λ = {eigenvalues[i]:.3f}')
    
plt.xlabel('x')
plt.ylabel('Displacement')
plt.title('Eigenmodes')
plt.legend()
plt.grid(True)
plt.show()