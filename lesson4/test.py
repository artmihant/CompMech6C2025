import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def rhs(t, u):

    x, y, z = u

    a_1 = 5
    b_1 = 2 # 2..6.2
    a_2 = 0.1
    b_2 = 2.0
    d_1 = 0.4
    d_2 = 0.01

    cons_1x = a_1*x/(1 + b_1*x)
    cons_2y = a_2*y/(1 + b_2*y)

    dx = x*(1 - x) - cons_1x*y
    dy = cons_1x*y - cons_2y*z - d_1*y
    dz = cons_2y*z - d_2*z

    return [dx, dy, dz]

u0 = [0.5, 0.3, 0.2]
t_span = (0.0, 2000.0)

sol = solve_ivp(rhs, t_span, u0,
                method="RK45", rtol=1e-9, atol=1e-12, max_step=0.1, dense_output=True)

t = np.linspace(0, 100, 10000)  # после транзиента
x, y, z = sol.sol(t)

fig, axs = plt.subplots(2, 2, figsize=(8,6))
axs[0,0].plot(t, y, lw=0.5); axs[0,0].set_title("y(t) (кролики)")
axs[0,1].plot(y, z, lw=0.3); axs[0,1].set_xlabel("y"); axs[0,1].set_ylabel("z")
axs[1,0].plot(x, y, lw=0.3); axs[1,0].set_xlabel("x"); axs[1,0].set_ylabel("y")
axs[1,1].plot(x, z, lw=0.3); axs[1,1].set_xlabel("x"); axs[1,1].set_ylabel("z")
plt.tight_layout(); plt.show()