import math
import numpy as np
import matplotlib.pyplot as plt

# PHYSICS

l = 2.0
theta0 = math.pi/100
omega0 = 0
g = 9.81
m = 1

def d(s):
    theta, omega = s
    return np.array([omega, - g/l * np.sin(theta)])

# NUMERIC

time = 10.0
dt = 0.02
t = np.arange(0, time, dt)
N = len(t)

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

# Initial state vector: [theta, omega]
s_0 = np.array([theta0, omega0])

# Analytical trajectory
theta_an = theta0*np.cos(np.sqrt(g/l)*t) + (omega0/np.sqrt(g/l))*np.sin(np.sqrt(g/l)*t)
omega_an = -theta0*np.sqrt(g/l)*np.sin(np.sqrt(g/l)*t)+omega0*np.cos(np.sqrt(g/l)*t)


# Initialize arrays
s_Euler = np.zeros((N, 2), dtype=np.float32)
s_Euler[0] = s_0
s_RK4 = np.zeros((N, 2), dtype=np.float32)
s_RK4[0] = s_0

def energy(theta, omega):
    U = m*g*l*(1 - np.cos(theta))
    K = 0.5*m*(l**2)*(omega**2)
    return U + K

# Integration loop
for i in range(1, N):
    s_Euler[i] = explicit_euler_step(s_Euler[i - 1])
    s_RK4[i] = rk4_step(s_RK4[i - 1])

theta_Euler = s_Euler[:, 0]
omega_Euler = s_Euler[:, 1]

theta_RK4 = s_RK4[:, 0]
omega_RK4 = s_RK4[:, 1]

energy_Euler = energy(theta_Euler, omega_Euler)
energy_RK4 = energy(theta_RK4, omega_RK4)
energy_an = energy(theta_an, omega_an)

max_theta_Euler = np.max(np.abs(theta_Euler - theta_an))
max_theta_RK4 = np.max(np.abs(theta_RK4 - theta_an))

energy_error_Euler = np.abs(energy_Euler[-1] - energy_an[-1]) / energy_an[-1] * 100
energy_error_RK4 = np.abs(energy_RK4[-1] - energy_an[-1]) / energy_an[-1] * 100

plt.subplot(2, 2, 1)
plt.plot(t, theta_Euler, label='Метод Эйлера')
plt.plot(t, theta_RK4, label='RK4')
plt.plot(t, theta_an, label='Аналитическое решение', linestyle='--')
plt.xlabel('Время t, c')
plt.ylabel(r'$\theta(t)$, рад')
plt.legend()
plt.title('Траектория')

plt.subplot(2, 2, 2)
plt.plot(theta_Euler, omega_Euler, label='Метод Эйлера')
plt.plot(theta_RK4, omega_RK4, label='RK4', lw=3)
plt.plot(theta_an, omega_an, label='Аналитическое решение', linestyle = '--')
plt.xlabel(r'Угол $\theta$, рад')
plt.ylabel(r'Угловая скорость $\omega$, рад/с')
plt.legend()
plt.title('Фазовый портрет')

plt.subplot(2, 2, 3)
plt.plot(t, energy_Euler, label='Метод Эйлера')
plt.plot(t, energy_RK4, label='RK4')
plt.plot(t, energy_an, label='Аналитическое решение', linestyle ='--')
plt.xlabel('Время t, с')
plt.ylabel('Энергия E, Дж')
plt.legend()
plt.title('Полная энергия')

plt.subplot(2, 2, 4)
plt.axis('off')  

text_content = f'Оценка скорости накопления ошибки:\n\n'
text_content += f'Макс. отклонение угла:\n'
text_content += f'• Метод Эйлера: {np.degrees(max_theta_Euler):.3f} град\n'
text_content += f'• Метод RK4: {np.degrees(max_theta_RK4):.3f} град\n\n'
text_content += f'Макс. отклонение энергии:\n'
text_content += f'• Метод Эйлера: {energy_error_Euler:.2f}%\n'
text_content += f'• Метод RK4: {energy_error_RK4:.2f}%'

plt.text(0.05, 0.95, text_content, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', 
         alpha=0.8, edgecolor='blue'))

plt.tight_layout()
plt.show()