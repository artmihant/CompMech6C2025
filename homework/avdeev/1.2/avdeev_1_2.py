import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.stats import linregress

# PHYSICS
v0 = 10
alpha = math.radians(45.0)
g = 9.81

flight_range = v0 * v0 * math.sin(2 * alpha) / g
vx0 = v0 * math.cos(alpha)
vy0 = v0 * math.sin(alpha)

# NUMERICS
nsteps = 100
dx = flight_range / nsteps
dt = dx / vx0

# PREPROCESSING
x = np.linspace(0, flight_range, nsteps + 1)
y_an = x * math.tan(alpha) - g * x * x / (2 * (v0 * math.cos(alpha)) ** 2)
y_Euler = np.zeros(nsteps + 1)
y_RK4 = np.zeros(nsteps + 1)
vy = np.zeros(nsteps + 1)
vy[0] = vy0

# NUMERICAL SIMULATION
for i in range(nsteps):
    y_Euler[i+1] = y_Euler[i] + vy[i] * dt
    k1 = vy[i]
    k2 = vy[i] - 0.5 * g * dt
    k3 = vy[i] - 0.5 * g * dt
    k4 = vy[i] - g * dt
    y_RK4[i+1] = y_RK4[i] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6.0
    vy[i+1] = vy[i] - g * dt

def energy(y, vx, vy):
    return (vx**2 + vy**2)/2 + g*y

energy_exact = np.full(nsteps + 1, energy(0, vx0, vy0))
energy_Euler = energy(y_Euler, np.full(nsteps + 1, vx0), vy)
energy_RK4 = energy(y_RK4, np.full(nsteps + 1, vx0), vy)

# FIGURES
fig1, ax1 = plt.subplots(2, 1)
fig1.tight_layout()
ax1[0].grid(True)
ax1[1].grid(True)
ax1[0].axis('equal')
ax1[0].set_title('Сравнение численной и аналитической траектории')
traject_an = ax1[0].plot(x, y_an, c='black', lw=2)[0]
traject_Euler = ax1[0].plot(x[:1], y_Euler[:1], '--', c='blue', lw=2)[0]
traject_RK4 = ax1[0].plot(x[:1], y_RK4[:1], '--', c='red', lw=2)[0]
ax1[0].legend(['Exact', 'Euler', 'RK4'])
ax1[0].set_xlabel('x')
ax1[0].set_ylabel('y')

def init_anim():
    traject_an.set_data(x, y_an)
    traject_Euler.set_data(x[:1], y_Euler[:1])
    traject_RK4.set_data(x[:1], y_RK4[:1])
    return (traject_an, traject_Euler, traject_RK4)

def loop_anim(i):
    traject_Euler.set_data(x[:i+2], y_Euler[:i+2])
    traject_RK4.set_data(x[:i+2], y_RK4[:i+2])
    return (traject_an, traject_Euler, traject_RK4)

ani = anim.FuncAnimation(fig=fig1, func=loop_anim, init_func=init_anim,
    frames=nsteps, interval=50, repeat=False)

ax1[1].set_title('Графики энергии для точного и численных решений')
ax1[1].plot(x, energy_exact, c='black')
ax1[1].plot(x, energy_Euler, '--', c='blue')
ax1[1].plot(x, energy_RK4, '--', c='red')
ax1[1].legend(['Exact', 'Euler', 'RK4'])
ax1[1].set_xlabel('x')
ax1[1].set_ylabel('energy')

error_x_Euler = np.abs(y_Euler - y_an)
error_x_RK4 = np.abs(y_RK4 - y_an)
error_E_Euler = np.abs(energy_Euler - energy_exact)
error_E_RK4 = np.abs(energy_RK4 - energy_exact)

cumerror_x_Euler = np.cumsum(error_x_Euler)
cumerror_x_RK4 = np.cumsum(error_x_RK4)
cumerror_E_Euler = np.cumsum(error_E_Euler)
cumerror_E_RK4 = np.cumsum(error_E_RK4)

logx = np.log(x[1:])
logerror_x_Euler = np.log(cumerror_x_Euler[1:])
logerror_x_RK4 = np.log(cumerror_x_RK4[1:])
logerror_E_Euler = np.log(cumerror_E_Euler[1:])
logerror_E_RK4 = np.log(cumerror_E_RK4[1:])

slope_x_Euler, _, _, _, _ = linregress(logx, logerror_x_Euler)
slope_x_RK4, _, _, _, _ = linregress(logx, logerror_x_RK4)
slope_E_Euler, _, _, _, _ = linregress(logx, logerror_E_Euler)
slope_E_RK4, _, _, _, _ = linregress(logx, logerror_E_RK4)

fig2, ax2 = plt.subplots(2, 1)
ax2[0].grid(True)
ax2[1].grid(True)
ax2[0].loglog(x, cumerror_x_Euler, c='blue')
ax2[0].loglog(x, cumerror_x_RK4, c='red')
ax2[0].set_title("Накопленная ошибка по координате (логарифмическая шкала)")
ax2[0].legend(['Euler, p={:.2f}'.format(slope_x_Euler), 'RK4, p={:.2f}'.format(slope_x_RK4)])
ax2[1].loglog(x, cumerror_E_Euler, c='blue')
ax2[1].loglog(x, cumerror_E_RK4, c='red')
ax2[1].set_title("Накопленная ошибка по энергии (логарифмическая шкала)")
ax2[1].legend(['Euler, p={:.2f}'.format(slope_E_Euler), 'RK4, p={:.2f}'.format(slope_E_RK4)])
fig2.tight_layout()

plt.show()