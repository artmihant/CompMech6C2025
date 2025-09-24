import numpy as np
import matplotlib.pyplot as plt

# Параметры маятника
g = 9.81  # ускорение свободного падения (м/с^2)
L = 1.0  # длина маятника (м)
m = 1.0  # масса маятника (кг)

# Начальные условия
theta0 = 0.1  # начальный угол (рад) - малые колебания
omega0 = 0.0  # начальная угловая скорость (рад/с)

# Параметры интегрирования
t_start = 0.0  # начальное время (с)
t_end = 10.0  # конечное время (с)
dt = 0.01  # шаг времени (с)
N = int((t_end - t_start) / dt) + 1  # количество точек

# Создаем массивы для хранения результатов
time = np.linspace(t_start, t_end, N)  # массив времени
theta_euler = np.zeros(N)  # углы для метода Эйлера
omega_euler = np.zeros(N)  # угловые скорости для метода Эйлера
theta_rk4 = np.zeros(N)  # углы для метода RK4
omega_rk4 = np.zeros(N)  # угловые скорости для метода RK4

# Устанавливаем начальные условия
theta_euler[0] = theta0
omega_euler[0] = omega0
theta_rk4[0] = theta0
omega_rk4[0] = omega0


# Функция для вычисления производных (угловое ускорение)
def pendulum_derivatives(theta, omega):
    # Уравнение маятника: (d^2)θ/dt^2 + (g/L)sin(θ) = 0
    # Преобразуем в систему двух уравнений первого порядка:
    # dθ/dt = w
    # dw/dt = -(g/L)sin(θ)
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return dtheta_dt, domega_dt


# Метод Эйлера
for i in range(N - 1):
    # Вычисляем производные в текущей точке
    dtheta, domega = pendulum_derivatives(theta_euler[i], omega_euler[i])

    # Простейший метод Эйлера: y_{n+1} = y_n + dt * f(y_n)
    theta_euler[i + 1] = theta_euler[i] + dt * dtheta
    omega_euler[i + 1] = omega_euler[i] + dt * domega

# Метод Рунге-Кутты 4-го порядка (RK4)
for i in range(N - 1):
    # Текущие значения
    theta_current = theta_rk4[i]
    omega_current = omega_rk4[i]

    # Вычисляем коэффициенты k1
    k1_theta, k1_omega = pendulum_derivatives(theta_current, omega_current)

    # Вычисляем коэффициенты k2 (промежуточная точка)
    theta_temp = theta_current + 0.5 * dt * k1_theta
    omega_temp = omega_current + 0.5 * dt * k1_omega
    k2_theta, k2_omega = pendulum_derivatives(theta_temp, omega_temp)

    # Вычисляем коэффициенты k3 (другая промежуточная точка)
    theta_temp = theta_current + 0.5 * dt * k2_theta
    omega_temp = omega_current + 0.5 * dt * k2_omega
    k3_theta, k3_omega = pendulum_derivatives(theta_temp, omega_temp)

    # Вычисляем коэффициенты k4 (конечная точка)
    theta_temp = theta_current + dt * k3_theta
    omega_temp = omega_current + dt * k3_omega
    k4_theta, k4_omega = pendulum_derivatives(theta_temp, omega_temp)

    # Суммируем взвешенные коэффициенты
    theta_rk4[i + 1] = theta_current + (dt / 6.0) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
    omega_rk4[i + 1] = omega_current + (dt / 6.0) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)

# Аналитическое решение для малых углов (гармонический осциллятор)
omega_analytical = np.sqrt(g / L)  # собственная частота
theta_analytical = theta0 * np.cos(omega_analytical * time)  # аналитическое решение

# Вычисляем энергию для каждого метода
# Полная энергия = кинетическая + потенциальная
# E = 0.5*m*L^2w^2 + m*g*L*(1 - cos(θ))

# Энергия для метода Эйлера
kinetic_energy_euler = 0.5 * m * (L * omega_euler) ** 2
potential_energy_euler = m * g * L * (1 - np.cos(theta_euler))
total_energy_euler = kinetic_energy_euler + potential_energy_euler

# Энергия для метода RK4
kinetic_energy_rk4 = 0.5 * m * (L * omega_rk4) ** 2
potential_energy_rk4 = m * g * L * (1 - np.cos(theta_rk4))
total_energy_rk4 = kinetic_energy_rk4 + potential_energy_rk4

# Начальная энергия (для сравнения)
E0 = total_energy_euler[0]

# Вычисляем ошибки относительно аналитического решения
error_euler = np.abs(theta_euler - theta_analytical)
error_rk4 = np.abs(theta_rk4 - theta_analytical)

# Графики
plt.figure(figsize=(15, 10))

# График 1: Угол как функция времени
plt.subplot(2, 3, 1)
plt.plot(time, theta_euler, 'b-', label='Метод Эйлера', alpha=0.7)
plt.plot(time, theta_rk4, 'r-', label='Метод RK4', alpha=0.7)
plt.plot(time, theta_analytical, 'g--', label='Аналитическое решение', alpha=0.8)
plt.xlabel('Время (с)')
plt.ylabel('Угол (рад)')
plt.title('Угол маятника vs время')
plt.legend()
plt.grid(True)

# График 2: Фазовый портрет (угловая скорость vs угол)
plt.subplot(2, 3, 2)
plt.plot(theta_euler, omega_euler, 'b-', label='Метод Эйлера', alpha=0.7)
plt.plot(theta_rk4, omega_rk4, 'r-', label='Метод RK4', alpha=0.7)
plt.xlabel('Угол (рад)')
plt.ylabel('Угловая скорость (рад/с)')
plt.title('Фазовый портрет')
plt.legend()
plt.grid(True)

# График 3: Полная энергия
plt.subplot(2, 3, 3)
plt.plot(time, total_energy_euler, 'b-', label='Метод Эйлера', alpha=0.7)
plt.plot(time, total_energy_rk4, 'r-', label='Метод RK4', alpha=0.7)
plt.axhline(y=E0, color='g', linestyle='--', label='Начальная энергия', alpha=0.8)
plt.xlabel('Время (с)')
plt.ylabel('Энергия (Дж)')
plt.title('Полная энергия системы')
plt.legend()
plt.grid(True)

# График 4: Отклонение энергии от начального значения
plt.subplot(2, 3, 4)
energy_error_euler = np.abs(total_energy_euler - E0)
energy_error_rk4 = np.abs(total_energy_rk4 - E0)
plt.plot(time, energy_error_euler, 'b-', label='Метод Эйлера', alpha=0.7)
plt.plot(time, energy_error_rk4, 'r-', label='Метод RK4', alpha=0.7)
plt.xlabel('Время (с)')
plt.ylabel('Отклонение энергии (Дж)')
plt.title('Отклонение энергии от начального значения')
plt.legend()
plt.grid(True)
plt.yscale('log')  # лог. шкала

# График 5: Ошибка относительно аналитического решения
plt.subplot(2, 3, 5)
plt.plot(time, error_euler, 'b-', label='Метод Эйлера', alpha=0.7)
plt.plot(time, error_rk4, 'r-', label='Метод RK4', alpha=0.7)
plt.xlabel('Время (с)')
plt.ylabel('Ошибка (рад)')
plt.title('Ошибка относительно аналитического решения')
plt.legend()
plt.grid(True)
plt.yscale('log')  # лог. шкала

plt.tight_layout()
plt.show()

print(f"Максимальная ошибка Эйлера: {np.max(error_euler):.6f} рад")
print(f"Максимальная ошибка RK4: {np.max(error_rk4):.6f} рад")
print(f"Конечная ошибка Эйлера: {error_euler[-1]:.6f} рад")
print(f"Конечная ошибка RK4: {error_rk4[-1]:.6f} рад")
print(f"Отклонение энергии Эйлера: {energy_error_euler[-1]:.6f} Дж")





