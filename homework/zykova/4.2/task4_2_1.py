import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Hastings and Powell
def food_chain(t, state, params):
    x, y, z = state
    a1, b1, a2, b2, d1, d2 = params
    
    f1 = a1 * x / (1 + b1 * x)
    f2 = a2 * y / (1 + b2 * y)
    
    dxdt = x * (1 - x) - f1 * y
    dydt = f1 * y - f2 * z - d1 * y
    dzdt = f2 * z - d2 * z
    
    return [dxdt, dydt, dzdt]

params = [5.0, 3.0, 0.1, 2.0, 0.4, 0.01] # a1, b1, a2, b2, d1, d2
initial_conditions = [0.5, 0.1, 0.1] 

t_span = (0, 5000)
t_eval = np.linspace(1000, 5000, 20000) 
solution = solve_ivp(food_chain, t_span, initial_conditions, args=(params,), t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)

t = solution.t
x, y, z = solution.y

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(t, x, 'b-', linewidth=1)
plt.ylabel('X (жертва)')
plt.xlabel('Время')
plt.ylim(0, 1.2)
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(t, y, 'g-', linewidth=1)
plt.ylabel('Y (хищник)')
plt.xlabel('Время')
plt.ylim(0, 0.3)
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(t, z, 'r-', linewidth=1)
plt.ylabel('Z (суперхищник)')
plt.xlabel('Время')
plt.ylim(0, 12) 
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x, y, 'b-', linewidth=0.5, alpha=0.7)
plt.xlabel('X (жертва)')
plt.ylabel('Y (хищник)')
plt.title('Фазовый портрет X-Y')
plt.grid(True, alpha=0.3)
plt.xlim(0, 1.2)
plt.ylim(0, 0.3)

plt.subplot(1, 3, 2)
plt.plot(y, z, 'g-', linewidth=0.5, alpha=0.7)
plt.xlabel('Y (хищник)')
plt.ylabel('Z (суперхищник)')
plt.title('Фазовый портрет Y-Z')
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.3)
plt.ylim(0, 12)

plt.subplot(1, 3, 3)
plt.plot(x, z, 'r-', linewidth=0.5, alpha=0.7)
plt.xlabel('X (жертва)')
plt.ylabel('Z (суперхищник)')
plt.title('Фазовый портрет X-Z')
plt.grid(True, alpha=0.3)
plt.xlim(0, 1.2)
plt.ylim(0, 12)

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.viridis(np.linspace(0, 1, len(x)))

for i in range(len(x)-1): ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], color=colors[i], linewidth=0.5, alpha=0.6)

ax.set_xlabel('X (жертва)')
ax.set_ylabel('Y (хищник)')
ax.set_zlabel('Z (суперхищник)')
ax.set_title('Трехмерный фазовый портрет пищевой цепи')

plt.tight_layout()
plt.show()