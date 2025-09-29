import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

K, beta, delta, gamma = 1.0, 1.1, 0.5, 0.4
r0, a, omega = 1.0, 0.6, 0.9
T = 2*np.pi/omega

def r_t(t):
    return r0 * (1 + a*np.cos(omega*t))

def rhs(t, u):
    x, y = u
    rx = r_t(t) * x * (1 - x/K)
    xy = beta * x * y
    return [rx - xy, delta*xy - gamma*y]

u0 = [0.3, 0.2]
t_span = (0.0, 400*T)

sol = solve_ivp(rhs, t_span, u0,
                method='RK45', rtol=1e-9, atol=1e-12,
                dense_output=True, max_step=0.05)

# Сечение Пуанкаре: берем точки через целое число периодов после транзиента
t_samples = np.arange(200*T, t_span[1], T)  # пропустить первые 200 периодов
x_s, y_s = sol.sol(t_samples)

plt.figure(figsize=(5,5))
plt.plot(x_s, y_s, '.', ms=2)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Poincaré-сечение сезонами форсированной модели трава–кролики')
plt.tight_layout(); plt.show()