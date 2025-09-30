import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g = 9.81
m1 = 1.0
m2 = 1.0
L1 = 1.0
L2 = 1.0

constants  = (m1, m2, L1, L2, g)

def double_pendulum_e(t, y):
    (m1, m2, L1, L2, g) = constants 
    th1, w1, th2, w2 = y

    delta = th1 - th2
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    den = m1 + m2 * sin_delta**2

    a1 = (
        m2 * g * np.sin(th2) * cos_delta
        - m2 * sin_delta * (L1 * w1**2 * cos_delta + L2 * w2**2)
        - (m1 + m2) * g * np.sin(th1)
    ) / (L1 * den)

    a2 = (
        (m1 + m2) * (L1 * w1**2 * sin_delta - g * np.sin(th2) + g * np.sin(th1) * cos_delta)
        + m2 * L2 * w2**2 * sin_delta * cos_delta
    ) / (L2 * den)

    return np.array([w1, a1, w2, a2], dtype=float)


t0, tf = 0.0, 20.0

# нач углы/скорости 
y0 = np.array([math.pi/2, 0.0, math.pi/2, 0.0], dtype=float)

#использую указанную в задании scipy.integrate: solve_ivp
sol = solve_ivp(
    fun=double_pendulum_e,
    t_span=(t0, tf),
    y0=y0,
    method="RK45",
    rtol=1e-10,   # относ точность
    atol=1e-13,   # абс точнсоть
    max_step=0.001,  # макс шаг ограничим
    vectorized=False
)

t = sol.t
y = sol.y.T  

plt.figure()
plt.plot(t, y[:, 0], label="θ_1")
plt.plot(t, y[:, 2], label="θ_2")
plt.xlabel("Время (секунды)")
plt.ylabel("Угол (радианы)")
plt.title("Графики изменения углов (RK45)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()