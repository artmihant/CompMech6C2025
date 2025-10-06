import math
import numpy as np
import matplotlib.pyplot as plt

def equations_of_motion_double_pendulum(t, y, params):
    (m1, m2, L1, L2, g) = params
    th1, w1, th2, w2 = y
    delta = th2 - th1

    den1 = (m1 + m2) * L1 - m2 * L1 * math.cos(delta)**2
    den2 = (L2 / L1) * den1

    a1 = (m2 * L1 * w1**2 * math.sin(delta) * math.cos(delta) +
          m2 * g * math.sin(th2) * math.cos(delta) +
          m2 * L2 * w2**2 * math.sin(delta) -
          (m1 + m2) * g * math.sin(th1)) / den1

    a2 = (-m2 * L2 * w2**2 * math.sin(delta) * math.cos(delta) +
          (m1 + m2) * (g * math.sin(th1) * math.cos(delta) -
                       L1 * w1**2 * math.sin(delta) -
                       g * math.sin(th2))) / den2

    return np.array([w1, a1, w2, a2], dtype=float)

def rk4(f, t0, y0, dt, n_steps, params):
    ts = np.empty(n_steps + 1)
    ys = np.empty((n_steps + 1, len(y0)))
    ts[0] = t0
    ys[0] = y0
    t = t0
    y = np.array(y0, dtype=float)

    for i in range(1, n_steps + 1):
        k1 = f(t, y, params)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, params)
        k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, params)
        k4 = f(t + dt, y + dt * k3, params)
        y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t = t + dt
        ts[i] = t
        ys[i] = y

    return ts, ys
    
# физические параметры
g = 9.8
m1, m2 = 1.0, 1.0
L1, L2 = 1.0, 1.0
params = (m1, m2, L1, L2, g)

# дискретизация времени
t0, tf = 0.0, 20.0
dt = 0.001
n_steps = int((tf - t0) / dt)

# углы
y0 = np.array([math.pi/2, 0.0, math.pi/2, 0.0], dtype=float)

# наше численное решение
t_rk4, y_rk4 = rk4(equations_of_motion_double_pendulum, t0, y0, dt, n_steps, params)

# графики углов
plt.figure()
plt.plot(t_rk4, y_rk4[:, 0], label="theta1 (rk4)")
plt.plot(t_rk4, y_rk4[:, 2], label="theta2 (rk4)")
plt.xlabel("время")
plt.ylabel("угол")
plt.title("Графики изменения углов")
plt.legend()
plt.minorticks_on()                 # доп деления
plt.grid(which="major", alpha=0.8)  # основная сетка
plt.grid(which="minor", alpha=0.3, linestyle=":")  #еще мелкая сетка
plt.tight_layout()
plt.show()
