import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры системы Лоренца
sigma = 10.0  # параметр σ
rho = 28.0  # параметр ρ
beta = 8.0 / 3.0  # параметр β

# Начальные условия для первой траектории
x0 = 1.0  # начальное значение x
y0 = 1.0  # начальное значение y
z0 = 1.0  # начальное значение z

# Начальные условия для второй траектории (немного отличаются)
x0_perturbed = 1.0001  # начальное значение x с небольшим возмущением
y0_perturbed = 1.0001  # начальное значение y с небольшим возмущением
z0_perturbed = 1.0001  # начальное значение z с небольшим возмущением

# Параметры интегрирования
t_start = 0.0  # начальное время
t_end = 40.0  # конечное время
dt = 0.01  # шаг времени
N = int((t_end - t_start) / dt) + 1  # количество точек
t_eval = np.linspace(t_start, t_end, N)  # массив времени для вычислений


# Функция, определяющая систему уравнений Лоренца
def lorenz_system(t, state, sigma, rho, beta):
    # Распаковываем переменные состояния
    x, y, z = state

    # Вычисляем производные по времени
    dx_dt = sigma * (y - x)  # уравнение для x: dx/dt = σ(y - x)
    dy_dt = x * (rho - z) - y  # уравнение для y: dy/dt = x(ρ - z) - y
    dz_dt = x * y - beta * z  # уравнение для z: dz/dt = xy - βz

    # Возвращаем производные в виде массива
    return [dx_dt, dy_dt, dz_dt]


# Решаем систему для первой траектории
solution1 = solve_ivp(lorenz_system,
                      [t_start, t_end],
                      [x0, y0, z0],
                      args=(sigma, rho, beta),
                      t_eval=t_eval,
                      method='RK45')

# Решаем систему для второй траектории (с возмущенными начальными условиями)
solution2 = solve_ivp(lorenz_system,
                      [t_start, t_end],
                      [x0_perturbed, y0_perturbed, z0_perturbed],
                      args=(sigma, rho, beta),
                      t_eval=t_eval,
                      method='RK45')

# Извлекаем решения для первой траектории
x1 = solution1.y[0]  # координата x первой траектории
y1 = solution1.y[1]  # координата y первой траектории
z1 = solution1.y[2]  # координата z первой траектории

# Извлекаем решения для второй траектории
x2 = solution2.y[0]  # координата x второй траектории
y2 = solution2.y[1]  # координата y второй траектории
z2 = solution2.y[2]  # координата z второй траектории

# Вычисляем разницу между траекториями
diff_x = np.abs(x1 - x2)  # разница по координате x
diff_y = np.abs(y1 - y2)  # разница по координате y
diff_z = np.abs(z1 - z2)  # разница по координате z

# Вычисляем евклидово расстояние между траекториями
distance = np.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2)

# Создаем графики
plt.figure(figsize=(15, 12))

# График 1: 2D-фазовый портрет в плоскости XY
plt.subplot(3, 4, 1)
plt.plot(x1, y1, 'b-', linewidth=0.5, alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Фазовый портрет (XY плоскость)')
plt.grid(True)

# График 2: 2D-фазовый портрет в плоскости XZ
plt.subplot(3, 4, 2)
plt.plot(x1, z1, 'b-', linewidth=0.5, alpha=0.7)
plt.xlabel('x')
plt.ylabel('z')
plt.title('Фазовый портрет (XZ плоскость)')
plt.grid(True)

# График 3: 2D-фазовый портрет в плоскости YZ
plt.subplot(3, 4, 3)
plt.plot(y1, z1, 'b-', linewidth=0.5, alpha=0.7)
plt.xlabel('y')
plt.ylabel('z')
plt.title('Фазовый портрет (YZ плоскость)')
plt.grid(True)

# График 4: Координата x как функция времени
plt.subplot(3, 4, 4)
plt.plot(t_eval, x1, 'b-', linewidth=0.5, alpha=0.7, label='Траектория 1')
plt.plot(t_eval, x2, 'r-', linewidth=0.5, alpha=0.7, label='Траектория 2')
plt.xlabel('Время')
plt.ylabel('x')
plt.title('Координата x vs время')
plt.legend()
plt.grid(True)

# График 5: Координата y как функция времени
plt.subplot(3, 4, 5)
plt.plot(t_eval, y1, 'b-', linewidth=0.5, alpha=0.7, label='Траектория 1')
plt.plot(t_eval, y2, 'r-', linewidth=0.5, alpha=0.7, label='Траектория 2')
plt.xlabel('Время')
plt.ylabel('y')
plt.title('Координата y vs время')
plt.legend()
plt.grid(True)

# График 6: Координата z как функция времени
plt.subplot(3, 4, 6)
plt.plot(t_eval, z1, 'b-', linewidth=0.5, alpha=0.7, label='Траектория 1')
plt.plot(t_eval, z2, 'r-', linewidth=0.5, alpha=0.7, label='Траектория 2')
plt.xlabel('Время')
plt.ylabel('z')
plt.title('Координата z vs время')
plt.legend()
plt.grid(True)

# График 7: Разница по координате x между траекториями
plt.subplot(3, 4, 7)
plt.plot(t_eval, diff_x, 'g-', linewidth=0.5, alpha=0.7)
plt.xlabel('Время')
plt.ylabel('Разница по x')
plt.title('Разница по координате x')
plt.grid(True)
plt.yscale('log')  # логарифмическая шкала по y

# График 8: Разница по координате y между траекториями
plt.subplot(3, 4, 8)
plt.plot(t_eval, diff_y, 'g-', linewidth=0.5, alpha=0.7)
plt.xlabel('Время')
plt.ylabel('Разница по y')
plt.title('Разница по координате y')
plt.grid(True)
plt.yscale('log')  # логарифмическая шкала по y

# График 9: Разница по координате z между траекториями
plt.subplot(3, 4, 9)
plt.plot(t_eval, diff_z, 'g-', linewidth=0.5, alpha=0.7)
plt.xlabel('Время')
plt.ylabel('Разница по z')
plt.title('Разница по координате z')
plt.grid(True)
plt.yscale('log')  # логарифмическая шкала по y

# График 10: Евклидово расстояние между траекториями
plt.subplot(3, 4, 10)
plt.plot(t_eval, distance, 'm-', linewidth=0.5, alpha=0.7)
plt.xlabel('Время')
plt.ylabel('Расстояние')
plt.title('Евклидово расстояние между траекториями')
plt.grid(True)
plt.yscale('log')  # логарифмическая шкала по y

# График 11: 3D-фазовый портрет (аттрактор Лоренца)
from mpl_toolkits.mplot3d import Axes3D

ax = plt.subplot(3, 4, 11, projection='3d')
ax.plot(x1, y1, z1, 'b-', linewidth=0.5, alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D аттрактор Лоренца')

plt.tight_layout()
plt.show()

# Выводим информацию о чувствительности к начальным условиям
print(f"Начальное расстояние между траекториями: {distance[0]:.6f}")
print(f"Конечное расстояние между траекториями: {distance[-1]:.6f}")
print(f"Максимальное расстояние между траекториями: {np.max(distance):.6f}")
print(f"Отношение конечного к начальному расстоянию: {distance[-1] / distance[0]:.6f}")