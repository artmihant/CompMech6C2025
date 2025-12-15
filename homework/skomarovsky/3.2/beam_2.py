import numpy as np
import matplotlib.pyplot as plt

# Solve second-order ODE for phi - Kirchhoff theory
def solve_phi_equation(phi_prime_0, EJ, L, F):
    """
    Solve the second-order differential equation:
     φ''(s) = -A1·cos(φ)
     A1 = F/k3 = F/EJ
    """
    
    # Convert to first-order system: [phi, phi']
    sol_0 = np.array((0.0, phi_prime_0), dtype=np.float64)
    
    s_0 = 0.0
    s_1 = L
    N = 300
    h = (s_1 - s_0) / N
    s = np.linspace(s_0, s_1, N + 1)
    
    sol = np.zeros((N + 1, 2), dtype=np.float64)
    sol[0] = sol_0
    
    # Parameters for equation (17)
    omega = F / EJ
    # For Kirchhoff theory (inextensible), simplified equation
    # φ''(s) = -ω·cos(φ) where ω = F/EJ
    # For downward bending, we need positive ω·cos(φ)
    A1 = omega  
    
    def f(state, s_i):
        phi = state[0]
        phi_prime = state[1]
        
        # φ' = φ'
        # φ'' = A1·cos(φ)
        dphi_ds = phi_prime
        dphi_prime_ds = A1 * np.cos(phi)
        
        return np.array((dphi_ds, dphi_prime_ds))
    
    # RK4 integration
    for i in range(N):
        s_i = s[i]
        d1 = h * f(sol[i], s_i)
        d2 = h * f(sol[i] + d1/2.0, s_i + h/2.0)
        d3 = h * f(sol[i] + d2/2.0, s_i + h/2.0)
        d4 = h * f(sol[i] + d3, s_i + h)
        sol[i+1] = sol[i] + (d1 + 2.0*d2 + 2.0*d3 + d4) / 6.0
    
    return s, sol

# Reconstruct x(s) and y(s) from phi(s) solution
def reconstruct_coordinates(s, phi_sol):
    """
    Given phi(s), integrate to get x(s) and y(s):
    x'(s) = cos(phi(s))
    y'(s) = sin(phi(s))
    """
    N = len(s)
    x = np.zeros(N)
    y = np.zeros(N)
    
    for i in range(1, N):
        ds = s[i] - s[i-1]
        phi_avg = (phi_sol[i] + phi_sol[i-1]) / 2.0
        
        x[i] = x[i-1] + ds * np.cos(phi_avg)
        y[i] = y[i-1] + ds * np.sin(phi_avg)
    
    return x, y

# Bisection method (shooting) to find phi'(0) such that phi'(L) = 0
def shooting_method_bisection(EJ, L, F):
    """
    Boundary value problem:
    φ(0) = 0, φ'(L) = 0
    
    Find φ'(0) using shooting method with bisection
    """
    
    print("\n=== Kirchhoff Theory - Shooting Method (Bisection) ===")
    print(f"Boundary conditions: φ(0) = 0, φ'(L) = 0")
    print("Finding φ'(0) using bisection method...")
    
    # Initial bracket for phi'(0)
    # Lower bound: small positive value
    # Upper bound: larger positive value (estimate based on linear theory)
    omega = F / EJ
    alpha_high = 0.0
    alpha_low = -2.0 * np.sqrt(omega * L)  # Rough estimate
    
    max_iter = 100
    tolerance = 1e-10
    
    iterations_data = []
    
    # Evaluate boundary condition at bracket endpoints
    s, sol_low = solve_phi_equation(alpha_low, EJ, L, F)
    f_low = sol_low[-1, 1]  # φ'(L) at lower bound
    
    s, sol_high = solve_phi_equation(alpha_high, EJ, L, F)
    f_high = sol_high[-1, 1]  # φ'(L) at upper bound
    
    print(f"Initial bracket: α ∈ [{alpha_low:.6e}, {alpha_high:.6e}]")
    print(f"f(α_low) = {f_low:.6e}, f(α_high) = {f_high:.6e}")
    
    # Check if bracket is valid
    if f_low * f_high > 0:
        print("WARNING: Initial bracket does not contain root. Adjusting...")
        # Try to find a valid bracket
        if f_low > 0 and f_high > 0:
            # Both positive, need to increase upper bound
            alpha_high *= 2.0
            s, sol_high = solve_phi_equation(alpha_high, EJ, L, F)
            f_high = sol_high[-1, 1]
        print(f"Adjusted bracket: α ∈ [{alpha_low:.6e}, {alpha_high:.6e}]")
        print(f"f(α_low) = {f_low:.6e}, f(α_high) = {f_high:.6e}")
    
    for iter_count in range(max_iter):
        # Bisection step
        alpha_mid = (alpha_low + alpha_high) / 2.0
        
        # Solve with midpoint
        s, sol_mid = solve_phi_equation(alpha_mid, EJ, L, F)
        f_mid = sol_mid[-1, 1]  # φ'(L)
        
        iterations_data.append({
            'iteration': iter_count,
            'phi_prime_0': alpha_mid,
            'phi_solution': sol_mid[:, 0],
            'error': abs(f_mid),
            'bracket_width': alpha_high - alpha_low
        })
        
        if iter_count % 5 == 0:
            print(f"Iter {iter_count:3d}: α = {alpha_mid:12.6e}, "
                  f"f(α) = {f_mid:12.6e}, bracket = {alpha_high - alpha_low:12.6e}")
        
        # Check convergence
        if abs(f_mid) < tolerance or (alpha_high - alpha_low) < tolerance:
            break
        
        # Update bracket
        if f_low * f_mid < 0:
            # Root is in [alpha_low, alpha_mid]
            alpha_high = alpha_mid
            f_high = f_mid
        else:
            # Root is in [alpha_mid, alpha_high]
            alpha_low = alpha_mid
            f_low = f_mid
    
    print(f"\nConverged in {iter_count+1} iterations")
    print(f"Final φ'(0) = {alpha_mid:.6e}")
    print(f"φ'(L) = {sol_mid[-1, 1]:.6e} (should be ~0)")
    print(f"φ(L) = {np.degrees(sol_mid[-1, 0]):.3f}°")
    
    # Reconstruct coordinates
    x, y = reconstruct_coordinates(s, sol_mid[:, 0])
    
    return iterations_data, s, sol_mid, x, y

# Newton's method (shooting) to find phi'(0) such that phi'(L) = 0
def shooting_method_newton(EJ, L, F):
    """
    Boundary value problem:
    φ(0) = 0, φ'(L) = 0
    
    Find φ'(0) using shooting method with Newton's method
    """
    
    print("\n=== Kirchhoff Theory - Shooting Method (Newton) ===")
    print(f"Boundary conditions: φ(0) = 0, φ'(L) = 0")
    print("Finding φ'(0) using Newton's method...")
    
    # Initial guess for phi'(0)
    phi_prime_0 = 0.01  # Small positive value (beam bends down, phi increases initially)
    h_alpha = 1e-6  # Step for numerical derivative
    
    max_iter = 50
    tolerance = 1e-10
    
    iterations_data = []
    
    for iter_count in range(max_iter):
        # Solve with current guess
        s, sol_0 = solve_phi_equation(phi_prime_0, EJ, L, F)
        s, sol_p = solve_phi_equation(phi_prime_0 + h_alpha, EJ, L, F)
        
        # Boundary condition error: φ'(L) should be 0
        f_0 = sol_0[-1, 1]  # φ'(L)
        f_p = sol_p[-1, 1]
        
        # Numerical derivative
        df_dphi_prime_0 = (f_p - f_0) / h_alpha
        
        # Newton step
        delta = -f_0 / df_dphi_prime_0
        phi_prime_0 = phi_prime_0 + delta
        
        iterations_data.append({
            'iteration': iter_count,
            'phi_prime_0': phi_prime_0,
            'phi_solution': sol_0[:, 0],
            'error': abs(f_0)
        })
        
        if iter_count % 5 == 0:
            print(f"Iter {iter_count:3d}: φ'(0) = {phi_prime_0:12.6e}, error |φ'(L)| = {abs(f_0):12.6e}")
        
        if abs(f_0) < tolerance or abs(delta) < tolerance:
            break
    
    print(f"\nConverged in {iter_count+1} iterations")
    print(f"Final φ'(0) = {phi_prime_0:.6e}")
    print(f"φ'(L) = {sol_0[-1, 1]:.6e} (should be ~0)")
    print(f"φ(L) = {np.degrees(sol_0[-1, 0]):.3f}°")
    
    # Reconstruct coordinates
    x, y = reconstruct_coordinates(s, sol_0[:, 0])
    
    return iterations_data, s, sol_0, x, y

# Linear theory for comparison
def solve_linear_theory(EJ, L, F, verbose=True):
    """
    Linear theory: cantilever with free end
    φ(0) = 0, φ'(L) = 0 => M(L) = 0
    
    For vertical force at free end:
    M(x) = -F·(L-x)
    φ'(x) = M(x)/EJ = -F·(L-x)/EJ
    φ(x) = -F·(L·x - x²/2)/EJ
    w'(x) = φ(x)
    w(x) = -F·(L·x²/2 - x³/6)/EJ
    """
    
    N = 300
    x = np.linspace(0, L, N+1)
    
    phi = -F * (L * x - x**2 / 2.0) / EJ
    w = -F * (L * x**2 / 2.0 - x**3 / 6.0) / EJ
    
    if verbose:
        print("\n=== Linear Theory ===")
        print(f"End deflection: w(L) = {w[-1]*1000:.3f} mm")
        print(f"End rotation: φ(L) = {np.degrees(phi[-1]):.3f}°")
    
    return x, w, phi


# Main simulation
if __name__ == "__main__":
    # Material and geometric properties
    E = 2.1e11  # Young's modulus (Pa) - steel
    b = 0.02    # Width (m)
    h = 0.005   # Height (m)
    I = (b * h**3) / 12  # Moment of inertia
    EJ = E * I  # k3 in equations
    L = 1.0     # Length (m)
    
    # Applied dead load at free end
    F = 150.0  # Force (N) - positive = downward
    
    print(f"=== Cantilever Beam - Kirchhoff Theory ===")
    print(f"Length: L = {L} m")
    print(f"Cross-section: {b*1000} mm x {h*1000} mm")
    print(f"Bending stiffness: EI = k3 = {EJ:.2e} N·m²")
    print(f"Dead load: F = {F} N (vertical)")
    print(f"\nBoundary conditions: φ(0) = 0, φ'(L) = 0 (free end)")
    
    # Linear theory
    x_linear, w_linear, phi_linear = solve_linear_theory(EJ, L, F)
    
    # Nonlinear theory - Shooting method (Newton)
    iterations_newton, s_newton, phi_sol_newton, x_newton, y_newton = shooting_method_newton(EJ, L, F)
    
    # Nonlinear theory - Shooting method (Bisection)
    iterations_bisection, s_bisection, phi_sol_bisection, x_bisection, y_bisection = shooting_method_bisection(EJ, L, F)
    
    # Verify beam length (using Newton solution)
    arc_length = 0.0
    for i in range(1, len(x_newton)):
        dx = x_newton[i] - x_newton[i-1]
        dy = y_newton[i] - y_newton[i-1]
        arc_length += np.sqrt(dx**2 + dy**2)
    
    # Compute load-deflection curves for various forces
    print("\n=== Computing Load-Deflection Curves ===")
    F_values = np.linspace(5, 250, 20)
    deflections_linear = []
    deflections_newton = []
    deflections_bisection = []
    
    for F_test in F_values:
        print(f"Computing for F = {F_test:.1f} N...", end='\r')
        
        # Linear theory (suppress output)
        x_lin, w_lin, phi_lin = solve_linear_theory(EJ, L, F_test, verbose=False)
        deflections_linear.append(abs(w_lin[-1]) * 1000)
        
        # Newton method (suppress output)
        try:
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            _, _, _, _, y_newt = shooting_method_newton(EJ, L, F_test)
            deflections_newton.append(abs(y_newt[-1]) * 1000)
            
            _, _, _, _, y_bisect = shooting_method_bisection(EJ, L, F_test)
            deflections_bisection.append(abs(y_bisect[-1]) * 1000)
            
            sys.stdout = old_stdout
        except:
            sys.stdout = old_stdout
            deflections_newton.append(np.nan)
            deflections_bisection.append(np.nan)
    
    print(f"\nLoad-deflection curves computed for {len(F_values)} load cases.")
    
    # Plotting
    fig = plt.figure(figsize=(20, 5))
    
    # Plot 1: Deformed beam shape
    ax1 = plt.subplot(1, 4, 1)
    
    # Fixed support
    support_height = max(abs(w_linear[-1]), abs(y_newton[-1])) * 1000 * 0.15
    ax1.plot([0, 0], [-support_height, support_height], 'k-', linewidth=4)
    ax1.plot([-5, -5], [-support_height*0.8, support_height*0.8], 'k-', linewidth=2)
    
    # Undeformed beam
    ax1.plot([0, L*1000], [0, 0], 'k--', alpha=0.5, linewidth=1, label='Undeformed')
    
    # Force arrow at undeformed end
    arrow_length = max(30, abs(y_newton[-1])*1000*0.15)
    ax1.arrow(L*1000, 0, 0, arrow_length,
             head_width=10, head_length=10, fc='red', ec='red', linewidth=2, alpha=0.8)
    ax1.text(L*1000 - 500, arrow_length/10 - 100, f'F = {F} N',
             fontsize=9, color='red', va='center')
    
    # Deformed shapes (flip y down)
    ax1.plot(x_linear*1000, -w_linear*1000, 'b-', linewidth=2.5, label='Linear theory')
    ax1.plot(x_newton*1000, -y_newton*1000, 'r-', linewidth=2.5, label='Newton')
    ax1.plot(x_bisection*1000, -y_bisection*1000, 'g--', linewidth=2, label='Bisection')
    
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    ax1.set_title('Deformed Beam Shape')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left')
    ax1.axis('equal')
    ax1.invert_yaxis()
    
    errors_newton = [data['error'] for data in iterations_newton]
    iters_newton = [data['iteration'] for data in iterations_newton]
    errors_bisection = [data['error'] for data in iterations_bisection]
    iters_bisection = [data['iteration'] for data in iterations_bisection]
    
    # Plot 2: Comparison of convergence rates
    ax2 = plt.subplot(1, 4, 2)
    ax2.semilogy(iters_newton, errors_newton, 'ro-', linewidth=2, markersize=4, label='Newton')
    ax2.semilogy(iters_bisection, errors_bisection, 'go-', linewidth=2, markersize=4, label='Bisection')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel("Boundary error |φ'(L)|")
    ax2.set_title('Convergence Comparison')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    
    # Plot 3: Comparison of end displacements
    ax4 = plt.subplot(1, 4, 3)
    categories = ['Linear\nTheory', 'Newton\nMethod', 'Bisection\nMethod']
    y_displacements = [abs(w_linear[-1]*1000), abs(y_newton[-1]*1000), abs(y_bisection[-1]*1000)]
    x_displacements = [0, abs(x_newton[-1] - L)*1000, abs(x_bisection[-1] - L)*1000]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    ax4.bar(x_pos - width/2, y_displacements, width, label='Vertical disp.', color='blue', alpha=0.7)
    ax4.bar(x_pos + width/2, x_displacements, width, label='Horiz. shortening', color='orange', alpha=0.7)
    
    ax4.set_ylabel('Displacement [mm]')
    ax4.set_title(f'End Displacements (F = {F} N)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 4: Load-Deflection curve at free end
    ax3 = plt.subplot(1, 4, 4)
    ax3.plot(F_values, deflections_linear, 'b-', linewidth=2.5, marker='o', markersize=5, label='Linear theory')
    ax3.plot(F_values, deflections_newton, 'r-', linewidth=2.5, marker='s', markersize=5, label='Newton (Nonlinear)')
    ax3.plot(F_values, deflections_bisection, 'g--', linewidth=2, marker='^', markersize=5, label='Bisection (Nonlinear)')
    ax3.set_xlabel('Applied Force F [N]')
    ax3.set_ylabel('End Deflection w(L) [mm]')
    ax3.set_title('Load-Deflection Curve at Free End')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    
    # Print summary
    print("\n=== Results Summary ===")
    print(f"\nLinear theory:")
    print(f"  End deflection: w(L) = {w_linear[-1]*1000:.3f} mm")
    print(f"  End rotation: φ(L) = {np.degrees(phi_linear[-1]):.3f}°")
    
    print(f"\nNonlinear theory (Newton):")
    print(f"  Iterations: {len(iterations_newton)}")
    print(f"  End position: x(L) = {x_newton[-1]*1000:.3f} mm")
    print(f"  End deflection: y(L) = {y_newton[-1]*1000:.3f} mm")
    print(f"  End rotation: φ(L) = {np.degrees(phi_sol_newton[-1, 0]):.3f}°")
    print(f"  Curvature at end: φ'(L) = {phi_sol_newton[-1, 1]:.6e}")
    print(f"  Initial parameter φ'(0) = {iterations_newton[-1]['phi_prime_0']:.6e}")
    
    print(f"\nNonlinear theory (Bisection):")
    print(f"  Iterations: {len(iterations_bisection)}")
    print(f"  End position: x(L) = {x_bisection[-1]*1000:.3f} mm")
    print(f"  End deflection: y(L) = {y_bisection[-1]*1000:.3f} mm")
    print(f"  End rotation: φ(L) = {np.degrees(phi_sol_bisection[-1, 0]):.3f}°")
    print(f"  Curvature at end: φ'(L) = {phi_sol_bisection[-1, 1]:.6e}")
    print(f"  Initial parameter φ'(0) = {iterations_bisection[-1]['phi_prime_0']:.6e}")
    
    print(f"\nBeam length verification:")
    print(f"  Original length: {L*1000:.3f} mm")
    print(f"  Actual arc length: {arc_length*1000:.3f} mm")
    print(f"  Error: {abs(L - arc_length)*1000:.6f} mm")
    
    plt.show()

