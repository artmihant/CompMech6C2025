import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

def lorenz(t, state, sigma=SIGMA, rho=RHO, beta=BETA):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def solve_trajectory(y0, t_span=(0.0, 50.0), n_points=20000, rtol=1e-8, atol=1e-10):

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        lorenz,
        t_span,
        y0,
        method="RK45",  
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        vectorized=False
    )
    if not sol.success:
        raise RuntimeError(f"Интегратор не справился: {sol.message}")
    return sol.t, sol.y

def main():
 
    t_span = (0.0, 50.0)
    n_points = 20000

    y0 = np.array([1.0, 1.0, 1.0], dtype=float)  # базовые начальные условия
    delta = 1e-8                                 # маленький сдвиг
    delta_vec = np.array([delta, 0.0, 0.0])      # смещаем только x
    y0_perturbed = y0 + delta_vec

    # Базовая траектория
    t, Y = solve_trajectory(y0, t_span=t_span, n_points=n_points)
    x, y, z = Y

    _, Y2 = solve_trajectory(y0_perturbed, t_span=t_span, n_points=n_points)
    x2, y2, z2 = Y2

    dx_dt = SIGMA * (y - x)
    dy_dt = x * (RHO - z) - y
    dz_dt = x * y - BETA * z

    delta_str = (
        f"Δy0 = [{delta_vec[0]:.1e}, {delta_vec[1]:.1e}, {delta_vec[2]:.1e}]"
    )

    fig1, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    axes[0].plot(x, dx_dt, lw=0.9)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("x'")
    axes[0].set_title("(x, x')")

    axes[1].plot(y, dy_dt, lw=0.9)
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("y'")
    axes[1].set_title("(y, y')")

    axes[2].plot(z, dz_dt, lw=0.9)
    axes[2].set_xlabel("z")
    axes[2].set_ylabel("z'")
    axes[2].set_title("(z, z')")

    fig1.suptitle(f"Аттрактор Лоренца — фазовые портреты производных\n{delta_str}", y=1.05)
    fig1.tight_layout()

    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes2[0].plot(t, x, lw=0.9)
    axes2[0].set_ylabel("x")

    axes2[1].plot(t, y, lw=0.9)
    axes2[1].set_ylabel("y")

    axes2[2].plot(t, z, lw=0.9)
    axes2[2].set_xlabel("t")
    axes2[2].set_ylabel("z")

    fig2.suptitle(f"Координаты во времени (базовая траектория)\n{delta_str}", y=1.02)
    fig2.tight_layout()

  
    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes3[0].plot(t, x,  lw=0.9, label="базовая")
    axes3[0].plot(t, x2, lw=0.9, alpha=0.85, label="смещённая")
    axes3[0].set_ylabel("x")
    axes3[0].legend(fontsize=8, loc="upper right")

    axes3[1].plot(t, y,  lw=0.9, label="базовая")
    axes3[1].plot(t, y2, lw=0.9, alpha=0.85, label="смещённая")
    axes3[1].set_ylabel("y")
    axes3[1].legend(fontsize=8, loc="upper right")

    axes3[2].plot(t, z,  lw=0.9, label="базовая")
    axes3[2].plot(t, z2, lw=0.9, alpha=0.85, label="смещённая")
    axes3[2].set_xlabel("t")
    axes3[2].set_ylabel("z")
    axes3[2].legend(fontsize=8, loc="upper right")

    fig3.suptitle(f"Сравнение координат двух близких решений во времени\n{delta_str}", y=1.02)
    fig3.tight_layout()

  
    diff_x = np.abs(x2 - x)
    diff_y = np.abs(y2 - y)
    diff_z = np.abs(z2 - z)

    fig4, axes4 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes4[0].plot(t, diff_x, lw=0.9)
    axes4[0].set_ylabel("|Δx(t)|")

    axes4[1].plot(t, diff_y, lw=0.9)
    axes4[1].set_ylabel("|Δy(t)|")

    axes4[2].plot(t, diff_z, lw=0.9)
    axes4[2].set_xlabel("t")
    axes4[2].set_ylabel("|Δz(t)|")

    fig4.suptitle(f"Расхождение двух траекторий по координатам\n{delta_str}", y=1.02)
    fig4.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
