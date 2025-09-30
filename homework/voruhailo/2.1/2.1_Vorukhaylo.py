import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры двойного маятника
g = 9.81  # ускорение свободного падения (м/с^2)
m1 = 1.0  # масса первого маятника (кг)
m2 = 1.0  # масса второго маятника (кг)
L1 = 1.0  # длина первого маятника (м)
L2 = 1.0  # длина второго маятника (м)

# Начальные условия для углов и скоростей
theta1_0 = np.pi / 2  # начальный угол первого маятника (рад)
theta2_0 = np.pi / 2  # начальный угол второго маятника (рад)
omega1_0 = 0.0  # начальная угловая скорость первого маятника (рад/с)
omega2_0 = 0.0  # начальная угловая скорость второго маятника (рад/с)

# Параметры интегрирования
t_start = 0.0  # начальное время (с)
t_end = 10.0  # конечное время (с)
dt = 0.01  # шаг времени для вывода результатов (с)
t_points = np.arange(t_start, t_end, dt)  # массив времени для вывода


# Функция для вычисления производных двойного маятника
def double_pendulum_derivatives(t, y):
    # y[0] = theta1, y[1] = theta2, y[2] = omega1, y[3] = omega2
    theta1, theta2, omega1, omega2 = y

    # Вычисляем промежуточные величины для упрощения формул
    delta_theta = theta2 - theta1
    sin_delta = np.sin(delta_theta)
    cos_delta = np.cos(delta_theta)

    # Вычисляем знаменатель для уравнений движения
    denom1 = m1 + m2 * sin_delta ** 2
    denom2 = L1 * denom1

    # Вычисляем угловые ускорения (вторые производные)
    # Уравнения движения двойного маятника (из лагранжевой механики)
    alpha1 = (m2 * g * np.sin(theta2) * cos_delta -
              m2 * sin_delta * (L1 * omega1 ** 2 * cos_delta + L2 * omega2 ** 2) -
              (m1 + m2) * g * np.sin(theta1)) / (L1 * denom1)

    alpha2 = ((m1 + m2) * (L1 * omega1 ** 2 * sin_delta - g * np.sin(theta2) + g * np.sin(theta1) * cos_delta) +
              m2 * L2 * omega2 ** 2 * sin_delta * cos_delta) / (L2 * denom1)

    # Возвращаем производные: [dtheta1/dt, dtheta2/dt, domega1/dt, domega2/dt]
    return [omega1, omega2, alpha1, alpha2]


# Начальный вектор состояния [theta1, theta2, omega1, omega2]
y0 = [theta1_0, theta2_0, omega1_0, omega2_0]

# Решаем систему ОДУ методом RK45 (встроенный в solve_ivp)
# Метод RK45
solution = solve_ivp(double_pendulum_derivatives,
                     [t_start, t_end],
                     y0,
                     method='RK45',
                     t_eval=t_points,
                     rtol=1e-6)

# Извлекаем результаты решения
time = solution.t  # массив времени
theta1 = solution.y[0]  # угол первого маятника
theta2 = solution.y[1]  # угол второго маятника
omega1 = solution.y[2]  # угловая скорость первого маятника
omega2 = solution.y[3]  # угловая скорость второго маятника

# Строим графики
plt.figure(figsize=(15, 10))

# График 1: Углы первого и второго маятников как функция времени
plt.subplot(2, 2, 1)
plt.plot(time, theta1, 'b-', label='Первый маятник (θ1)', alpha=0.8)
plt.plot(time, theta2, 'r-', label='Второй маятник (θ2)', alpha=0.8)
plt.xlabel('Время (с)')
plt.ylabel('Угол (рад)')
plt.title('Углы двойного маятника vs время')
plt.legend()
plt.grid(True)

# График 2: Угловые скорости первого и второго маятников
plt.subplot(2, 2, 2)
plt.plot(time, omega1, 'b-', label='Первый маятник (ω1)', alpha=0.8)
plt.plot(time, omega2, 'r-', label='Второй маятник (ω2)', alpha=0.8)
plt.xlabel('Время (с)')
plt.ylabel('Угловая скорость (рад/с)')
plt.title('Угловые скорости двойного маятника')
plt.legend()
plt.grid(True)

# График 3: Фазовый портрет первого маятника
plt.subplot(2, 2, 3)
plt.plot(theta1, omega1, 'b-', alpha=0.7)
plt.xlabel('Угол первого маятника (рад)')
plt.ylabel('Угловая скорость первого маятника (рад/с)')
plt.title('Фазовый портрет первого маятника')
plt.grid(True)

# График 4: Фазовый портрет второго маятника
plt.subplot(2, 2, 4)
plt.plot(theta2, omega2, 'r-', alpha=0.7)
plt.xlabel('Угол второго маятника (рад)')
plt.ylabel('Угловая скорость второго маятника (рад/с)')
plt.title('Фазовый портрет второго маятника')
plt.grid(True)

plt.tight_layout()
plt.show()

# График 5: Разность углов
plt.figure(figsize=(10, 6))
plt.plot(time, theta1 - theta2, 'g-', alpha=0.8)
plt.xlabel('Время (с)')
plt.ylabel('Разность углов θ1 - θ2 (рад)')
plt.title('Разность углов между маятниками')
plt.grid(True)
plt.show()


print(f"Время интегрирования: от {t_start} до {t_end} с")
print(f"Количество точек: {len(time)}")
print(f"Начальные углы: θ1 = {theta1_0:.3f} рад, θ2 = {theta2_0:.3f} рад")
print(f"Конечные углы: θ1 = {theta1[-1]:.3f} рад, θ2 = {theta2[-1]:.3f} рад")
print(f"Максимальный угол первого маятника: {np.max(np.abs(theta1)):.3f} рад")
print(f"Максимальный угол второго маятника: {np.max(np.abs(theta2)):.3f} рад")