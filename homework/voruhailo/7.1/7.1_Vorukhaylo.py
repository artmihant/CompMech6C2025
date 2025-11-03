import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Длина струны
L = 1.0
# Скорость распространения волны
c = 1.0

# Количество точек по пространству (чем больше, тем точнее)
Nx = 100
# Шаг по пространству
dx = L / (Nx - 1)

# Массив координат x для визуализации
x = np.linspace(0, L, Nx)

# Время моделирования
T_max = 2.0

# Число Куранта (CFL) - параметр для устойчивости схемы
CFL = 0.5
# Вычисляем шаг по времени из условия устойчивости
dt = CFL * dx / c

# Количество шагов по времени
Nt = int(T_max / dt) + 1

# Создаем массив для хранения решения u(x,t) в три момента времени:
# u_prev - предыдущий временной слой (n-1)
# u_curr - текущий временной слой (n)
# u_next - следующий временной слой (n+1)
u_prev = np.zeros(Nx)
u_curr = np.zeros(Nx)
u_next = np.zeros(Nx)


# Функция начальной формы струны - "щипок" в центре
def initial_shape(x):
    # Создаем треугольный профиль с максимумом в центре
    peak_pos = L / 2  # Центр струны
    peak_height = 1.0  # Высота щипка

    # Для каждой точки x вычисляем высоту
    result = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] <= peak_pos:
            # Возрастающая часть треугольника (от 0 до центра)
            result[i] = (x[i] / peak_pos) * peak_height
        else:
            # Убывающая часть треугольника (от центра до L)
            result[i] = ((L - x[i]) / (L - peak_pos)) * peak_height

    return result


# Применяем начальную форму струны
u_curr = initial_shape(x)

# Начальная скорость равна нулю (g(x) = 0), поэтому u_prev вычисляем специальным образом
# Для второго порядка точности по времени при нулевой начальной скорости:
u_prev = u_curr.copy()

# Коэффициент для разностной схемы
r = (c * dt / dx) ** 2

# Создаем фигуру для анимации
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, u_curr, 'b-', linewidth=2)
ax.set_xlim(0, L)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('Позиция x')
ax.set_ylabel('Отклонение u(x,t)')
ax.set_title(f'Колебания струны (CFL = {CFL})')
ax.grid(True)


# Функция инициализации анимации
def init():
    line.set_ydata(u_curr)
    return line,


# Функция обновления анимации для каждого кадра
def update(frame):
    global u_prev, u_curr, u_next

    # Вычисляем следующий временной слой по схеме "крест"
    for i in range(1, Nx - 1):
        # Явная схема: u_next[i] = 2*u_curr[i] - u_prev[i] + r*(u_curr[i+1] - 2*u_curr[i] + u_curr[i-1])
        u_next[i] = 2 * u_curr[i] - u_prev[i] + r * (u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1])

    # Граничные условия: закрепленные концы
    u_next[0] = 0
    u_next[-1] = 0

    # Обновляем массивы для следующего шага
    u_prev = u_curr.copy()
    u_curr = u_next.copy()

    # Обновляем линию на графике
    line.set_ydata(u_curr)

    return line,


# Создаем анимацию
anim = FuncAnimation(fig, update, frames=Nt, init_func=init, blit=True, interval=20)

plt.show()

# Функция для исследования влияния CFL
def investigate_CFL():
    # Разные значения CFL для исследования
    CFL_values = [0.1, 0.5, 1.0, 1.1]

    # Создаем подграфики
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, CFL_test in enumerate(CFL_values):
        # Пересчитываем параметры для данного CFL
        dt_test = CFL_test * dx / c
        Nt_test = int(1.0 / dt_test) + 1  # Смотрим на меньшее время для сравнения

        # Инициализируем массивы
        u_prev_test = np.zeros(Nx)
        u_curr_test = initial_shape(x)
        u_next_test = np.zeros(Nx)
        u_prev_test = u_curr_test.copy()

        r_test = (c * dt_test / dx) ** 2

        # Выполняем несколько шагов по времени
        for n in range(Nt_test):
            for i in range(1, Nx - 1):
                u_next_test[i] = 2 * u_curr_test[i] - u_prev_test[i] + r_test * (
                            u_curr_test[i + 1] - 2 * u_curr_test[i] + u_curr_test[i - 1])

            u_next_test[0] = 0
            u_next_test[-1] = 0

            u_prev_test = u_curr_test.copy()
            u_curr_test = u_next_test.copy()

        # Рисуем результат
        axes[idx].plot(x, u_curr_test, 'r-', linewidth=2)
        axes[idx].set_xlim(0, L)
        axes[idx].set_ylim(-1.5, 1.5)
        axes[idx].set_xlabel('Позиция x')
        axes[idx].set_ylabel('Отклонение u(x,t)')
        axes[idx].set_title(f'CFL = {CFL_test}')
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()

print("Исследование влияния CFL на решение:")
print("CFL < 1: схема устойчива")
print("CFL = 1: оптимальная точность")
print("CFL > 1: схема неустойчива")
investigate_CFL()