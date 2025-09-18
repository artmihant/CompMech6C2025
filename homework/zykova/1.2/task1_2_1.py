import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PHYSICS

v0 = 100
alpha = math.radians(45.0)
g = 9.81

flight_range = v0 * v0 * math.sin(2 * alpha) / g
vx = v0 * math.cos(alpha)
vy = v0 * math.sin(alpha)

def d(s):
    x, y, vx, vy = s
    return np.array([vx, vy, 0, -g])

# NUMERIC

N = 50
dx = flight_range / N
dt = dx / vx

# INTEGRATION_METHODS

def explicit_euler_step(s):
    d1 = dt * d(s)
    return s + d1

def rk4_step(s):
    d1 = dt * d(s)
    d2 = dt * d(s + d1 / 2)
    d3 = dt * d(s + d2 / 2)
    d4 = dt * d(s + d3)
    return s + (d1 + 2 * d2 + 2 * d3 + d4) / 6

# PREPROCESSING

# Initial state vector: [x, y, vx, vy]
s_0 = np.array([0.0, 0.0, vx, vy])

# Analytical trajectory
x_an = np.linspace(0, flight_range, N + 1)
y_an = x_an * math.tan(alpha) - g * x_an * x_an / (2.0 * v0 * v0 * math.cos(alpha) ** 2)


# SIMULATION

# Initialize arrays
s_Euler = np.zeros((N + 1, 4), dtype=np.float32)
s_Euler[0] = s_0
s_RK4 = np.zeros((N + 1, 4), dtype=np.float32)
s_RK4[0] = s_0

# Integration loop
for i in range(1, N + 1):
    s_Euler[i] = explicit_euler_step(s_Euler[i - 1])
    if s_Euler[i][1] < 0.0:  
        s_Euler[i][1] = 0.0  
        break

for i in range(1, N + 1):
    s_RK4[i] = rk4_step(s_RK4[i - 1])
    if s_RK4[i][1] < 0.0:  
        s_RK4[i][1] = 0.0  
        break

x_Euler = s_Euler[:, 0]
y_Euler = s_Euler[:, 1]
vx_Euler = s_Euler[:, 2]
vy_Euler = s_Euler[:, 3]

x_RK4 = s_RK4[:, 0]
y_RK4 = s_RK4[:, 1]
vx_RK4 = s_RK4[:, 2]
vy_RK4 = s_RK4[:, 3]

def calculate_energy_and_momentum(y, vx, vy):
    kinetic_energy = 0.5 * (vx**2 + vy**2)
    potential_energy = g * y
    total_energy = kinetic_energy + potential_energy
    return total_energy


E_Euler = calculate_energy_and_momentum(y_Euler, vx_Euler, vy_Euler)
E_RK4 = calculate_energy_and_momentum(y_RK4, vx_RK4, vy_RK4)
E_an = (vx ** 2 + vy ** 2) * 0.5

# VISUALIZATION

euler_energy_error = abs(E_an - E_Euler[len(E_Euler) - 1])
euler_x_error = math.sqrt(abs(flight_range**2 - x_Euler[len(x_Euler)-1]**2))
euler_y_error = math.sqrt(abs(0.0 - y_Euler[len(y_Euler)-1]**2))

rk4_energy_error = abs(E_an - E_RK4[len(E_RK4) - 1])
rk4_x_error = math.sqrt(abs(flight_range**2 - x_RK4[len(x_RK4)-1]**2))
rk4_y_error = math.sqrt(abs(0.0 - y_RK4[len(y_RK4)-1]**2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(x_an, y_an, 'k--', label='Аналитическая траектория')
ax1.plot(x_Euler, y_Euler, label='Метод Эйлера')
ax1.plot(x_RK4, y_RK4, label='Метод RK4')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Сравнение траекторий')
ax1.legend(loc='upper left', frameon=False, fontsize=7)
ax1.axis('equal')

ax2.axhline(y=E_an, c='g', label='Аналитическая энергия')
ax2.plot(x_Euler, E_Euler, label='Метод Эйлера')
ax2.plot(x_RK4, E_RK4, label='Метод RK4')
ax2.set_xlabel('x')
ax2.set_ylabel('Энергия')
ax2.set_title('Сравнение энергий')
ax2.legend(loc='upper left', frameon=False, fontsize=7)

text_str_euler = (
    "Метод Эйлера:\n"
    f"Ошибка энергии: {euler_energy_error:.6f}\n"
    f"Ошибка координаты x: {euler_x_error:.6f}\n"
    f"Ошибка координаты y: {euler_y_error:.6f}"
)

text_str_rk4 = (
    "RK4:\n"
    f"Ошибка энергии: {rk4_energy_error:.6f}\n"
    f"Ошибка координаты x: {rk4_x_error:.6f}\n"
    f"Ошибка координаты y: {rk4_y_error:.6f}"
)

plt.figtext(0.02, 0.03, text_str_euler, fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
plt.figtext(0.3, 0.03, text_str_rk4, fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))

plt.tight_layout(rect=(0, 0.3, 1, 0.95)) 
plt.show()