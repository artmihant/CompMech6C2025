import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PHYSICS
# единица длины = 1 а е
# единица времени = 1 год
# единица массы = масса Солнца

a_Earth = 1.0 # большая полуось в а.е.
e_Earth = 0.017 # эксцентриситет орбиты
T_Earth = 1 # орбитальный период в годах

# гравитационный параметр mu = 4 * pi**2 * a**3 / T**2
GM = 4 * np.pi**2 * a_Earth**3 / T_Earth**2 # G * M_sun

def d(s):
    x, y, vx, vy = s
    r = math.sqrt(x**2 + y**2)
    return np.array([vx, vy, -GM*x/r**3, -GM*y/r**3])

# INITIAL CONDITIONS

x0, y0 = (1-e_Earth)*a_Earth, 0
vx0, vy0 = 0, math.sqrt(GM*(1+e_Earth)/(a_Earth*(1-e_Earth)))

# Analytical trajectory
# r(θ) = a(1-e^2) / (1 + e*cos(θ))
def analytic_solution(a, e):
    theta = np.linspace(0, 2*np.pi, 1000)
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

x_analytic, y_analytic = analytic_solution(a_Earth, e_Earth)

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

# NUMERIC

t_begin = 0
t_end = T_Earth * 3 
N = 40000      # nsteps_num
dt = (t_end - t_begin) / N

# PREPROCESSING

time = np.linspace(t_begin, t_end, N+1)

# Initial state vector: [x, y, vx, vy]
s_0_Euler = np.array([x0, y0, vx0, vy0])
s_0_RK4 = np.array([x0, y0, vx0, vy0])

# SIMULATION

# Initialize arrays
s_Euler = np.zeros((N + 1, 4), dtype=np.float32)
s_Euler[0] = s_0_Euler
s_RK4 = np.zeros((N + 1, 4), dtype=np.float32)
s_RK4[0] = s_0_RK4

# Integration loop
for i in range(1, N + 1):
    s_Euler[i] = explicit_euler_step(s_Euler[i - 1])
    s_RK4[i] = rk4_step(s_RK4[i - 1])

# Extract x, y coordinates and vx, vy
x_Euler = s_Euler[:, 0]
y_Euler = s_Euler[:, 1]
vx_Euler = s_Euler[:, 2]
vy_Euler = s_Euler[:, 3]

x_RK4 = s_RK4[:, 0]
y_RK4 = s_RK4[:, 1]
vx_RK4 = s_RK4[:, 2]
vy_RK4 = s_RK4[:, 3]

def calculate_energy_and_momentum(x, y, vx, vy, a):
    r = np.sqrt(x**2 + y**2)
    # орбитальная скорость в квадрате
    v_square = GM*(2/r - 1/a)
    kinetic_energy = 0.5 * v_square
    potential_energy = - GM / r
    total_energy = kinetic_energy + potential_energy
    angular_momentum = x * vy - y * vx
    return total_energy, angular_momentum

E_euler, L_euler = calculate_energy_and_momentum(x_Euler, x_Euler, vx_Euler, vy_Euler, a_Earth)
E_rk4, L_rk4 = calculate_energy_and_momentum(x_RK4, y_RK4, vx_RK4, vy_RK4, a_Earth)

# VISUALIZATION

plt.subplot(2, 2, 1)
plt.plot(x_analytic, y_analytic, 'k--', label='Аналитическая орбита')
plt.plot(x_Euler, y_Euler, label='Метод Эйлера')
plt.plot(x_RK4, y_RK4, label='Метод RK4')
plt.scatter([0], [0], c='yellow', s=1000)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5) 
plt.xlabel('x (а.е.)')
plt.ylabel('y (а.е.)')
plt.title('Сравнение орбит')
plt.legend(loc='upper left', frameon=False, fontsize=7)
plt.xlim(-1.05, 1.05)
plt.ylim(-1.05, 1.05)

plt.subplot(2, 2, 2)
plt.plot(time, E_euler, label='Метод Эйлера')
plt.plot(time, E_rk4, label='Метод RK4')
# E = -GM/(2a) - удельная орбитальная энергия
plt.axhline(y = -(GM) / (2*a_Earth), color='r', linestyle='--', label='Теоретическая энергия') 
plt.xlabel('Время (годы)')
plt.ylabel('E (J/kg)')
plt.title('Полная энергия')
plt.legend(loc='lower left', frameon=False, fontsize=7)

plt.subplot(2, 2, 3)
plt.plot(time, L_euler, label='Метод Эйлера')
plt.plot(time, L_rk4, label='Метод RK4')
plt.xlabel('Время (годы)')
plt.ylabel('L')
plt.title('Угловой момент')
plt.legend(loc='upper left', frameon=False, fontsize=7)

plt.subplot(2, 2, 4)
initial_energy_rk4 = E_rk4[0]
initial_energy_euler = E_euler[0]
error_euler = (E_euler - initial_energy_euler) / np.abs(initial_energy_euler)
error_rk4 = (E_rk4 - initial_energy_rk4) / np.abs(initial_energy_rk4)
plt.plot(time, error_euler, label='Метод Эйлера')
plt.plot(time, error_rk4, label='Метод RK4')
plt.xlabel('Время (годы)')
plt.ylabel('Относительная ошибка энергии')
plt.title('Накопление ошибки в энергии')
plt.legend(loc='lower left', frameon=False, fontsize=7)

plt.tight_layout()

plt.show()