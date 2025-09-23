import numpy as np
import matplotlib.pyplot as plt 

T_end = 10.0
N = 2000
t = np.linspace(0.0, T_end, N + 1)

theta0 = 0.08  
omega0 = 0.0 

g = 9.8
L = 1.0    
m = 1.0   

#эйлер
def euler_method(theta0, omega0, t, g, L):
    theta = np.zeros_like(t)
    omega = np.zeros_like(t)

    theta[0] = theta0
    omega[0] = omega0

    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        d_theta = omega[i]
        d_omega = -(g / L) * np.sin(theta[i])
        theta[i+1] = theta[i] + dt * d_theta
        omega[i+1] = omega[i] + dt * d_omega
    return theta, omega

#рунге-кутт4
def rk4_method(theta0, omega0, t, g, L):
    theta = np.zeros_like(t)
    omega = np.zeros_like(t)

    theta[0] = theta0
    omega[0] = omega0

    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        th = theta[i]
        om = omega[i]

        k1_th = om
        k1_om = -(g / L) * np.sin(th)
        
        th2 = th + 0.5 * dt * k1_th
        om2 = om + 0.5 * dt * k1_om
        k2_th = om2
        k2_om = -(g / L) * np.sin(th2)
        
        th3 = th + 0.5 * dt * k2_th
        om3 = om + 0.5 * dt * k2_om
        k3_th = om3
        k3_om = -(g / L) * np.sin(th3)
        
        th4 = th + dt * k3_th
        om4 = om + dt * k3_om
        k4_th = om4
        k4_om = -(g / L) * np.sin(th4)
      
        theta[i+1] = th + (dt/6.0)*(k1_th + 2*k2_th + 2*k3_th + k4_th)
        omega[i+1] = om + (dt/6.0)*(k1_om + 2*k2_om + 2*k3_om + k4_om)
    return theta, omega
#энергия
def energy(theta, omega, m, g, L):
    K = 0.5 * m * (L**2) * (omega**2)
    V = m * g * L * (1 - np.cos(theta))

    return K + V

def small_angle_solution(t, theta0, omega0, g, L):
    w = np.sqrt(g / L)
    A = theta0
    B = omega0 / w
    theta = A * np.cos(w * t) + B * np.sin(w * t)
    omega = -A * w * np.sin(w * t) + B * w * np.cos(w * t)

    return theta, omega

theta_eu, omega_eu = euler_method(theta0, omega0, t, g, L)
theta_rk, omega_rk = rk4_method(theta0, omega0, t, g, L)

theta_an, omega_an = small_angle_solution(t, theta0, omega0, g, L)

E_eu = energy(theta_eu, omega_eu, m, g, L)
E_rk = energy(theta_rk, omega_rk, m, g, L)
E_an = energy(theta_an, omega_an, m, g, L)

#графики
# 1)траектория θ(t) — аналитика + Эйлер + RK4
plt.figure(figsize=(10, 4))
plt.plot(t, theta_an, label='Аналитическое (малые углы)')
plt.plot(t, theta_eu, label='Эйлер')
plt.plot(t, theta_rk, label='Рунге–Кутта 4')
plt.xlabel('t, c'); plt.ylabel('θ, рад')
plt.title('Траектория θ(t)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()

# 2)фазовый портрет для разных начальных углов
ic_list = [0.1, 0.2, 0.3, 0.4, 0.5]  # рад
plt.figure(figsize=(6,5))
for th0 in ic_list:
    th, om = rk4_method(th0, 0.0, t, g, L)  
    plt.plot(th, om, label=f'θ0={th0:.1f} рад, ω0=0')
plt.xlabel('θ, рад'); plt.ylabel('ω, рад/с')
plt.title('Фазовый портрет: разные начальные условия')
plt.legend(ncol=2, fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 3)энергия E(t) — аналитика + Эйлер + RK4
plt.figure(figsize=(10, 4))
plt.plot(t, E_an, label='Аналитическое (малые углы)')
plt.plot(t, E_eu, label='Эйлер')
plt.plot(t, E_rk, label='Рунге–Кутта 4')
plt.xlabel('t, c'); plt.ylabel('E, Дж')
plt.title('Полная энергия E(t)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()

# 4)ошибки по углу
plt.figure(figsize=(10, 4))
plt.plot(t, np.abs(theta_eu - theta_an), label='|Эйлер−Аналитика|')
plt.plot(t, np.abs(theta_rk - theta_an), label='|RK4−Аналитика|')
plt.xlabel('t, c'); plt.ylabel('|Δθ|, рад')
plt.title('Абс ошибка по углу')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()

# 5)ошибки по энергии 
plt.figure(figsize=(10, 4))
plt.plot(t, np.abs(E_eu - E_an), label='|Эйлер−Аналитика|')
plt.plot(t, np.abs(E_rk - E_an), label='|RK4−Аналитика|')
plt.xlabel('t, c'); plt.ylabel('|ΔE|')
plt.title('Абс ошибка по энергии')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
