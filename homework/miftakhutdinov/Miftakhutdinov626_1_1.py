import numpy as np
import matplotlib.pyplot as plt

# параметры системы
g = 9.81
L = 1.0
m = 1.0

# дискретизация времени
T = 10.0
dt = 0.01
t = np.arange(0.0, T, dt)
n = len(t)

# начальные условия
theta0 = np.deg2rad(5.0)
omega0 = 0.0

# --- численные методы -------

def euler(theta0, omega0, t, g, L):
    theta = np.zeros_like(t)
    omega = np.zeros_like(t)
    theta[0] = theta0
    omega[0] = omega0
    for i in range(len(t) - 1):
        h = t[i+1] - t[i]
        dtheta = omega[i]
        domega = -(g / L) * np.sin(theta[i])
        theta[i+1] = theta[i] + h * dtheta
        omega[i+1] = omega[i] + h * domega
    return theta, omega

def rk4(theta0, omega0, t, g, L):
    theta = np.zeros_like(t)
    omega = np.zeros_like(t)
    theta[0] = theta0
    omega[0] = omega0
    for i in range(len(t) - 1):
        h = t[i+1] - t[i]
        th = theta[i]
        om = omega[i]
        # k1
        k1_th = om
        k1_om = -(g / L) * np.sin(th)
        # k2
        th2 = th + 0.5 * h * k1_th
        om2 = om + 0.5 * h * k1_om
        k2_th = om2
        k2_om = -(g / L) * np.sin(th2)
        # k3
        th3 = th + 0.5 * h * k2_th
        om3 = om + 0.5 * h * k2_om
        k3_th = om3
        k3_om = -(g / L) * np.sin(th3)
        # k4
        th4 = th + h * k3_th
        om4 = om + h * k3_om
        k4_th = om4
        k4_om = -(g / L) * np.sin(th4)
        # шаг
        theta[i+1] = th + (h/6.0)*(k1_th + 2*k2_th + 2*k3_th + k4_th)
        omega[i+1] = om + (h/6.0)*(k1_om + 2*k2_om + 2*k3_om + k4_om)
    return theta, omega

# энергия
def energy(theta, omega, m, g, L):
    K = 0.5 * m * (L**2) * (omega**2)
    V = m * g * L * (1 - np.cos(theta))
    return K + V

# аналитическое решение для малых углов
def an_solution(t, theta0, omega0, g, L):
    w = np.sqrt(g / L)
    A = theta0
    B = omega0 / w
    theta = A * np.cos(w * t) + B * np.sin(w * t)
    omega = -A * w * np.sin(w * t) + B * w * np.cos(w * t)
    return theta, omega

# расчёты
theta_eu, omega_eu = euler(theta0, omega0, t, g, L)
theta_rk, omega_rk = rk4(theta0, omega0, t, g, L)
theta_an, omega_an = an_solution(t, theta0, omega0, g, L)

E_eu = energy(theta_eu, omega_eu, m, g, L)
E_rk = energy(theta_rk, omega_rk, m, g, L)
E_an = energy(theta_an, omega_an, m, g, L)

ic_list = [np.deg2rad(5), np.deg2rad(10), np.deg2rad(20), np.deg2rad(29)]
plt.figure(figsize=(6,5))
for th0 in ic_list:
    th, om = rk4(th0, 0.0, t, g, L)
    deg0 = np.rad2deg(th0)  #обратно в градусы
    plt.plot(th, om, label=f'start: θ₀={deg0:.0f}°, ω₀=0')
plt.xlabel('θ, rad'); plt.ylabel('ω, rad/s')
plt.title('Phase portrait of pendulum')
plt.legend(ncol=2, fontsize=9)
plt.grid(True)
plt.tight_layout()


plt.figure(figsize=(10, 4))
plt.plot(t, E_an, label='Linear analytic ')
plt.plot(t, E_eu, label='Euler method')
plt.plot(t, E_rk, label='RK4 method')
plt.xlabel('t, s'); plt.ylabel('E, J')
plt.title('Total energy vs time')
plt.legend(); plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 4))
plt.plot(t, theta_an, label='Linear analytic')
plt.plot(t, theta_eu, label='Euler explicit')
plt.plot(t, theta_rk, label='RK4')
plt.xlabel('t, s'); plt.ylabel('θ, rad')
plt.title('Angle θ(t)')
plt.legend(); plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 4))
plt.plot(t, np.abs(E_eu - E_an), label='|ΔE|: Euler vs analytic')
plt.plot(t, np.abs(E_rk - E_an), label='|ΔE|: RK4 vs analytic')
plt.xlabel('t, s'); plt.ylabel('|ΔE|, J')
plt.title('Absolute energy error')
plt.legend(); plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 4))
plt.plot(t, np.abs(theta_eu - theta_an), label='|Δθ|: Euler vs analytic')
plt.plot(t, np.abs(theta_rk - theta_an), label='|Δθ|: RK4 vs analytic')
plt.xlabel('t, s'); plt.ylabel('|Δθ|, rad')
plt.title('Absolute angle error')
plt.legend(); plt.grid(True)
plt.tight_layout()

plt.show()
