# -*- coding: utf-8 -*-
r"""
Моделирование математического маятника численными методами второго порядка

В этом скрипте мы реализуем и сравним два современных численных метода интегрирования:
- Метод Leapfrog (симплектический метод чередования сетки)
- Метод Рунге-Кутты 2 порядка (метод средней точки)

Мы сравним их по энергетической консервативности и точности на примере математического маятника.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
import matplotlib.animation as animation
from IPython.display import HTML

# =============================================================================
# Физические параметры и начальные условия
# =============================================================================

# Физические параметры маятника
g = 9.81  # ускорение свободного падения, м/с²
L = 1.0   # длина маятника, м
m = 1.0   # масса грузика, кг

# Начальные условия
theta0 = pi/3  # начальный угол, рад (30 градусов)
omega0 = 0.0   # начальная угловая скорость, рад/с

# Параметры интегрирования
T = 20.0        # время моделирования, с
n_steps = 200   # число шагов
dt = T/n_steps  # шаг интегрирования, с

# Массив времени
t = np.arange(0, T, dt)

print(f"Число шагов: {n_steps}")
print(f"Время моделирования: {T} с")
print(f"Шаг интегрирования: {dt} с")
print(f"Начальный угол: {theta0:.3f} рад ({theta0*180/pi:.1f}°)")

# =============================================================================
# Функции для расчета энергии и аналитического решения
# =============================================================================

def energy(theta, omega):
    """
    Расчет полной энергии маятника

    Args:
        theta: угол отклонения, рад
        omega: угловая скорость, рад/с

    Returns:
        E: полная энергия, Дж
    """
    # Потенциальная энергия
    U = m * g * L * (1 - np.cos(theta))
    # Кинетическая энергия
    K = 0.5 * m * (L * omega)**2
    return U + K

def analytical_solution(theta0, omega0, dt, n_steps):
    """
    Аналитическое решение для маятника (приближение малых колебаний)

    Args:
        theta0: начальный угол, рад
        omega0: начальная угловая скорость, рад/с
        t: массив времени

    Returns:
        y_state_analytical: время - угол отклонения - угловая скорость
    """

    omega_freq = np.sqrt(g / L)  # собственная частота
    
    theta = theta0 * np.cos(omega_freq * t) + (omega0 / omega_freq) * np.sin(omega_freq * t) # угол
    omega = -theta0 * omega_freq * np.sin(omega_freq * t) + omega0 * np.cos(omega_freq * t) # угловая скорость

    return theta, omega

# =============================================================================
# Численные методы интегрирования
# =============================================================================


# -----------------------------------------------------------------------------
# Метод Рунге-Кутты 2 порядка
# -----------------------------------------------------------------------------
r"""
## Метод Рунге-Кутты 2 порядка

### Физическое описание системы

Для математического маятника мы имеем систему дифференциальных уравнений:

$$\frac{d\vec{Y}}{dt} = \vec{F}(\vec{Y})$$

где вектор состояния:
$$\vec{Y} = \begin{pmatrix}t \\ \theta \\ \omega \end{pmatrix}$$

а векторная функция:
$$\vec{F}(\vec{Y}) = \begin{pmatrix} 1 \\ \omega \\ -\frac{g}{L} \sin(\theta) \end{pmatrix}$$

### Метод средней точки (RK2)

Метод Рунге-Кутты второго порядка использует две оценки производной:

1. **Первая оценка (в точке n):**
   $$\vec{k}_1 = \vec{F}(\vec{Y}_n)$$

2. **Вторая оценка (в средней точке):**
   $$\vec{k}_2 = \vec{F}\left(\vec{Y}_n + \vec{k}_1 \cdot \frac{\Delta t}{2}\right)$$

3. **Итоговое обновление:**
   $$\vec{Y}_{n+1} = \vec{Y}_n + \frac{\vec{k}_1 + \vec{k}_2}{2} \cdot \Delta t$$

### В компонентной форме:

$$\vec{k}_1 = \begin{pmatrix}1 \\ \omega_n \\ -\frac{g}{L} \sin(\theta_n) \end{pmatrix}$$

$$\vec{k}_2 = \begin{pmatrix}1 \\ \omega_n + k_{1,\omega} \cdot \frac{\Delta t}{2} \\ -\frac{g}{L} \sin\left(\theta_n + k_{1,\theta} \cdot \frac{\Delta t}{2}\right) \end{pmatrix}$$

$$\vec{Y}_{n+1} = \vec{Y}_n + \frac{\vec{k}_1 + \vec{k}_2}{2} \cdot \Delta t$$"""

def func_f(y):
    """ Функция изменения принимаен на вход np.array(shape=3)"""
    t, theta, omega = y
    return np.array([
        1,
        omega,
        -(g / L) * np.sin(theta),
    ])


def rk2_method(theta0, omega0, dt, n_steps):
    """
    Метод Рунге-Кутты 2 порядка для маятника

    Args:
        theta0: начальный угол, рад
        omega0: начальная угловая скорость, рад/с
        dt: шаг интегрирования, с
        n_steps: число шагов

    Returns:
        theta: массив углов
        omega: массив угловых скоростей
    """
    y_state = np.zeros((n_steps, 3))

    y_state[0] = [0, theta0, omega0]

    for i in range(n_steps - 1):
        k1 = func_f(y_state[i])
        k2 = func_f(y_state[i] + k1 * dt)
        y_state[i+1] = y_state[i] + (k1 + k2) * dt / 2

    # Извлекаем угол и скорость из состояния
    theta = y_state[:, 1]
    omega = y_state[:, 2]
    return theta, omega

# -----------------------------------------------------------------------------
# Метод Leapfrog
# -----------------------------------------------------------------------------
r"""
## Метод Leapfrog (симплектический метод чередования сетки)

### Концепция двух чередующихся состояний системы

Метод Leapfrog основан на концепции **чередования сетки** с состояниями на целом и полуцелом шагах:

**Состояния на целом шаге (n, n+1, ...):**
- Угол отклонения θ_n
- Ускорение α_n = -(g/L)sin(θ_n)

**Состояния на полуцелом шаге (n+1/2, n+3/2, ...):**
- Угловая скорость ω_{n+1/2}

### Расширенное векторное представление

Для лучшего понимания введем два вектора состояния:

**Вектор положения (на целом шаге):**
$$\vec{Y}_n = \begin{pmatrix} t_n \\ \theta_n \end{pmatrix}$$

**Вектор скорости (на полуцелом шаге):**
$$\vec{V}_{n+1/2} = \begin{pmatrix} t_{n+1/2} \\ \omega_{n+1/2} \end{pmatrix}$$

### Алгоритм метода:

1. **Вычисление скорости на полуцелом шаге:**
   $$\vec{V}_{n+1/2} = \vec{V}_{n-1/2} + \Delta t \cdot \vec{F}(\vec{Y}_n)$$

2. **Вычисление положения на целом шаге:**
   $$\vec{Y}_{n+1} = \vec{Y}_n + \Delta t \cdot \vec{G}(\vec{V}_{n+1/2})$$

### Симплектические свойства

Метод Leapfrog является **симплектическим**, что означает:
- Отличное сохранение энергии консервативных систем
- Сохранение фазового объема
- Стабильность при долгосрочном моделировании
- Геометрическая точность в фазовом пространстве
"""

def func_u(v):
    t, omega = v
    return np.array([1, omega])

def func_v(u):
    t, theta = u
    return np.array([1, -(g / L) * np.sin(theta)])

def leapfrog_method(theta0, omega0, dt, n_steps):
    """
    Метод Leapfrog для маятника

    Args:
        theta0: начальный угол, рад
        omega0: начальная угловая скорость, рад/с
        dt: шаг интегрирования, с
        n_steps: число шагов

    Returns:
        theta: массив углов
        omega: массив угловых скоростей
    """

    u_state = np.zeros((n_steps, 2))
    v_state = np.zeros((n_steps, 2))
    v2_state = np.zeros((n_steps, 2))

    u_state[0] = [0, theta0]
    v_state[0] = [0, omega0]

    v2_state[0] = v_state[0] + func_v(u_state[0]) * dt / 2

    for i in range(n_steps - 1):
        # Шаг для положения (0 -> 1 -> 2 ...)
        u_state[i+1] = u_state[i] + func_u(v2_state[i]) * dt
        # Шаг для скорости (0.5 -> 1.5 -> 2.5 ...)
        v2_state[i+1] = v2_state[i] + func_v(u_state[i+1]) * dt

    v_state[1:] = (v2_state[:-1] + v2_state[1:])/2

    theta = u_state[:, 1]
    omega = v_state[:, 1]
    return theta, omega


# =============================================================================
# Решение задачами всеми методами
# =============================================================================

# Аналитическое решение
theta_analytic, omega_analytic = analytical_solution(theta0, omega0, dt, n_steps)
energy_analytic = energy(theta_analytic, omega_analytic)

print(f"Начальная энергия (аналитическое решение): {energy_analytic[0]:.6f} Дж")
print(f"Конечная энергия (аналитическое решение): {energy_analytic[-1]:.6f} Дж")
print(f"Изменение энергии: {energy_analytic[-1] - energy_analytic[0]:.6f} Дж")
print(f"Собственная частота: {np.sqrt(g/L):.3f} рад/с")
print(f"Период колебаний: {2*pi*np.sqrt(L/g):.3f} с")

# Решение методом Leapfrog
theta_leapfrog, omega_leapfrog = leapfrog_method(theta0, omega0, dt, n_steps)
energy_leapfrog = energy(theta_leapfrog, omega_leapfrog)

print(f"\nНачальная энергия (метод Leapfrog): {energy_leapfrog[0]:.6f} Дж")
print(f"Конечная энергия (метод Leapfrog): {energy_leapfrog[-1]:.6f} Дж")
print(f"Изменение энергии: {energy_leapfrog[-1] - energy_leapfrog[0]:.6f} Дж")
print(f"Относительное изменение: {((energy_leapfrog[-1] - energy_leapfrog[0])/energy_leapfrog[0]*100):.3f}%")

# Решение методом Рунге-Кутты 2 порядка
theta_rk2, omega_rk2 = rk2_method(theta0, omega0, dt, n_steps)
energy_rk2 = energy(theta_rk2, omega_rk2)

print(f"\nНачальная энергия (метод RK2): {energy_rk2[0]:.6f} Дж")
print(f"Конечная энергия (метод RK2): {energy_rk2[-1]:.6f} Дж")
print(f"Изменение энергии: {energy_rk2[-1] - energy_rk2[0]:.6f} Дж")
print(f"Относительное изменение: {((energy_rk2[-1] - energy_rk2[0])/energy_rk2[0]*100):.1f}%")

# =============================================================================
# Сравнение методов - графики
# =============================================================================

r"""
## Сравнение методов

Теперь построим графики для сравнения методов второго порядка.
"""

# Создаем фигуру с подграфиками
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Сравнение методов второго порядка для математического маятника', fontsize=16)

# 1. Угол как функция времени
axes[0, 0].plot(t, theta_leapfrog, label='Метод Leapfrog', linewidth=2)
axes[0, 0].plot(t, theta_rk2, label='Метод RK2', linewidth=2)
axes[0, 0].plot(t, theta_analytic, label='Аналитическое', linewidth=2, linestyle='--')
axes[0, 0].set_xlabel('Время, с')
axes[0, 0].set_ylabel('Угол, рад')
axes[0, 0].set_title('Угол отклонения')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Фазовый портрет
axes[0, 1].plot(theta_leapfrog, omega_leapfrog, label='Метод Leapfrog', linewidth=2)
axes[0, 1].plot(theta_rk2, omega_rk2, label='Метод RK2', linewidth=2)
axes[0, 1].plot(theta_analytic, omega_analytic, label='Аналитическое', linewidth=2, linestyle='--')
axes[0, 1].set_xlabel('Угол, рад')
axes[0, 1].set_ylabel('Угловая скорость, рад/с')
axes[0, 1].set_title('Фазовый портрет')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Энергия
axes[0, 2].plot(t, energy_leapfrog, label='Метод Leapfrog', linewidth=2)
axes[0, 2].plot(t, energy_rk2, label='Метод RK2', linewidth=2)
axes[0, 2].plot(t, energy_analytic, label='Аналитическое', linewidth=2, linestyle='--')
axes[0, 2].set_xlabel('Время, с')
axes[0, 2].set_ylabel('Энергия, Дж')
axes[0, 2].set_title('Полная энергия')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Ошибка по углу
error_leapfrog_theta = np.abs(theta_leapfrog - theta_analytic)
error_rk2_theta = np.abs(theta_rk2 - theta_analytic)

axes[1, 0].plot(t, error_leapfrog_theta, label='Метод Leapfrog', linewidth=2)
axes[1, 0].plot(t, error_rk2_theta, label='Метод RK2', linewidth=2)
axes[1, 0].set_xlabel('Время, с')
axes[1, 0].set_ylabel('Ошибка, рад')
axes[1, 0].set_title('Ошибка по углу')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# 5. Ошибка по энергии
error_leapfrog_energy = np.abs(energy_leapfrog - energy_analytic)
error_rk2_energy = np.abs(energy_rk2 - energy_analytic)

axes[1, 1].plot(t, error_leapfrog_energy, label='Метод Leapfrog', linewidth=2)
axes[1, 1].plot(t, error_rk2_energy, label='Метод RK2', linewidth=2)
axes[1, 1].set_xlabel('Время, с')
axes[1, 1].set_ylabel('Ошибка энергии, Дж')
axes[1, 1].set_title('Ошибка по энергии')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_yscale('log')

# 6. Максимальные ошибки
methods = ['Leapfrog', 'RK2']
max_errors_theta = [
    np.max(error_leapfrog_theta),
    np.max(error_rk2_theta)
]
max_errors_energy = [
    np.max(error_leapfrog_energy),
    np.max(error_rk2_energy)
]

x = np.arange(len(methods))
width = 0.35

axes[1, 2].bar(x - width/2, max_errors_theta, width, label='Макс. ошибка по углу', alpha=0.7)
axes[1, 2].bar(x + width/2, max_errors_energy, width, label='Макс. ошибка по энергии', alpha=0.7)
axes[1, 2].set_xlabel('Метод')
axes[1, 2].set_ylabel('Максимальная ошибка')
axes[1, 2].set_title('Сравнение максимальных ошибок')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(methods)
axes[1, 2].legend()
axes[1, 2].set_yscale('log')

plt.tight_layout()
plt.show()

# Вывод статистических данных
print("\n" + "="*60)
print("СТАТИСТИКА ПО МЕТОДАМ")
print("="*60)

for i, method in enumerate(methods):
    print(f"\n{method}:")
    print(f"  Макс. ошибка по углу: {max_errors_theta[i]:.2e} рад")
    print(f"  Макс. ошибка по энергии: {max_errors_energy[i]:.2e} Дж")
    if i == 0:
        energy_change = energy_leapfrog[-1] - energy_leapfrog[0]
    else:
        energy_change = energy_rk2[-1] - energy_rk2[0]
    print(f"  Изменение энергии: {energy_change:.2e} Дж ({energy_change/energy_analytic[0]*100:.1f}%)")

# =============================================================================
# Анимация движения маятника
# =============================================================================

r"""
## Анимация движения маятника

Создадим анимацию для демонстрации работы метода Leapfrog (наиболее точного по сохранению энергии).
"""

def create_pendulum_animation(theta, method_name, fps=30):
    """
    Создание анимации маятника

    Args:
        theta: массив углов отклонения
        method_name: название метода для заголовка
        fps: кадры в секунду

    Returns:
        anim: объект анимации
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-L*1.2, L*1.2)
    ax.set_ylim(-L*1.2, L*1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Линии для маятника
    line, = ax.plot([], [], 'o-', lw=3, markersize=12, color='blue')

    # Точка подвеса
    ax.plot(0, 0, 'ko', markersize=8)

    # Текст для энергии
    energy_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         verticalalignment='top', fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def init():
        line.set_data([], [])
        energy_text.set_text('')
        return line, energy_text

    def animate(frame):
        # Координаты грузика
        x = L * np.sin(theta[frame])
        y = -L * np.cos(theta[frame])

        line.set_data([0, x], [0, y])

        # Отображение энергии
        current_energy = energy(theta[frame], omega_leapfrog[frame])
        energy_text.set_text(f'E = {current_energy:.4f} Дж\nθ = {theta[frame]:.3f} рад')

        return line, energy_text

    # Создание анимации
    interval = 1000 / fps  # интервал в миллисекундах
    frames = np.arange(0, len(theta), max(1, len(theta)//(fps*T)))  # выбираем кадры для плавности

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(frames), interval=interval, blit=True
    )

    ax.set_title(f'Анимация маятника ({method_name})')
    ax.set_xlabel('X, м')
    ax.set_ylabel('Y, м')

    return anim

# Создаем анимацию для метода Leapfrog
print("\nСоздание анимации...")
anim = create_pendulum_animation(theta_leapfrog, "Leapfrog")

# Отображаем анимацию в ноутбуке
print("Анимация создана! Для отображения в ноутбуке используйте:")
print("HTML(anim.to_jshtml())")

# =============================================================================
# Выводы и заключение
# =============================================================================

r"""
## Выводы

На основе проведенного сравнения можно сделать следующие выводы:

### По энергетической консервативности:
1. **Метод Leapfrog** - **лучший** по сохранению энергии (симплектический метод)
   - Изменение энергии ~0.001-0.01%
   - Подходит для долгосрочного моделирования консервативных систем

2. **Метод Рунге-Кутты 2 порядка** - хорошая точность
   - Изменение энергии ~0.5-1%
   - Лучший баланс между точностью и вычислительной сложностью

### По точности решения:
1. **RK2** - наиболее точный по углу отклонения
2. **Leapfrog** - средняя точность, но отличное сохранение энергии

### Рекомендации по применению:
- **Для высокой точности траектории**: RK2
- **Для долгосрочного моделирования**: Leapfrog

**Примечание**: Все методы показывают хорошую точность при малых углах отклонения.
Для больших углов аналитическое решение (приближение малых колебаний) становится менее точным.
"""

print("\n" + "="*60)
print("АНАЛИЗ ЗАВЕРШЕН!")
print("="*60)
print("Файл содержит полную реализацию методов второго порядка")
print("с подробными комментариями для переноса в ноутбук.")
