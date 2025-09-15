# -*- coding: utf-8 -*-
r"""
Моделирование затухающего гармонического осциллятора численными методами

В этом скрипте мы реализуем и сравним восемь численных методов интегрирования:
- Метод Leapfrog (симплектический, адаптированный для диссипативных систем)
- Метод Рунге-Кутты 4 порядка (явный одношаговый метод)
- Метод Адамса-Бэшфорта 4 порядка (явный многошаговый метод)
- Метод Адамса-Моултона 2 порядка (неявный многошаговый метод)
- Метод Адамса-Моултона 4 порядка (неявный многошаговый метод)

Мы сравним их по точности и эффективности на примере затухающего гармонического осциллятора.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
import matplotlib.animation as animation
from IPython.display import HTML

# =============================================================================
# Физические параметры и начальные условия
# =============================================================================

# -----------------------------------------------------------------------------
# Формула гармонического осциллятора:
# 
# Уравнение движения:
# $$ m\ddot{x} + b\dot{x} + kx = 0 $$
# 
# где:
# - $m$ — масса,
# - $b$ — коэффициент трения,
# - $k$ — жесткость пружины,
# - $x$ — смещение.
# -----------------------------------------------------------------------------


# Физические параметры гармонического осциллятора
m = 1.0   # масса, кг
k = 1000.0  # жесткость пружины, Н/м
b = 0.0  # коэффициент трения, кг/с

# Начальные условия
x0 = 0.5   # начальное смещение, м
v0 = 0.0   # начальная скорость, м/с

# Параметры интегрирования
T = 10.0          # время моделирования, с (уменьшили для жесткой системы)
dt = 0.01      # шаг интегрирования, с (уменьшили для стабильности Адамса)
n_steps = int(T/dt)   # число шагов

# Массив времени
t = np.arange(0, T, dt)

print(f"Число шагов: {n_steps}")
print(f"Время моделирования: {T} с")
print(f"Шаг интегрирования: {dt} с")
print(f"Начальное смещение: {x0:.3f} м")
print(f"Начальная скорость: {v0:.3f} м/с")
print(f"Масса: {m:.1f} кг")
print(f"Жесткость пружины: {k:.1f} Н/м")
print(f"Коэффициент трения: {b:.1f} кг/с")

# =============================================================================
# Функции для расчета энергии и аналитического решения
# =============================================================================

def energy(x, v):
    """
    Расчет полной энергии гармонического осциллятора

    Args:
        x: смещение, м
        v: скорость, м/с

    Returns:
        E: полная энергия, Дж
    """
    # Потенциальная энергия
    U = 0.5 * k * x**2
    # Кинетическая энергия
    K = 0.5 * m * v**2
    return U + K

def analytical_solution(x0, v0, dt, n_steps):
    """
    Аналитическое решение для затухающего гармонического осциллятора

    Args:
        x0: начальное смещение, м
        v0: начальная скорость, м/с
        dt: шаг времени
        n_steps: число шагов

    Returns:
        x_analytical: массив смещений
        v_analytical: массив скоростей
    """

    # Параметры затухающего осциллятора
    omega0 = np.sqrt(k / m)  # собственная частота
    gamma = b / (2 * m)      # коэффициент затухания

    # Определяем тип колебаний
    discriminant = omega0**2 - gamma**2

    t = np.arange(0, T, dt)

    if discriminant > 0:  # Недозатухающий режим
        omega = np.sqrt(discriminant)
        A = x0
        B = (v0 + gamma * x0) / omega

        x = np.exp(-gamma * t) * (A * np.cos(omega * t) + B * np.sin(omega * t))
        v = np.exp(-gamma * t) * ((-gamma * A - omega * B) * np.cos(omega * t) +
                                  (-gamma * B + omega * A) * np.sin(omega * t))

    elif discriminant < 0:  # Перезагужающий режим
        alpha = np.sqrt(-discriminant)
        A = x0
        B = (v0 + gamma * x0) / alpha

        x = np.exp(-gamma * t) * (A * np.cosh(alpha * t) + B * np.sinh(alpha * t))
        v = np.exp(-gamma * t) * ((-gamma * A - alpha * B) * np.cosh(alpha * t) +
                                  (-gamma * B + alpha * A) * np.sinh(alpha * t))

    else:  # Критическое затухание
        A = x0
        B = v0 + gamma * x0

        x = np.exp(-gamma * t) * (A + B * t)
        v = np.exp(-gamma * t) * (-gamma * A - gamma * B * t + B)

    return x, v

# =============================================================================
# Численные методы интегрирования
# =============================================================================


# -----------------------------------------------------------------------------
# Метод Рунге-Кутты 2 порядка
# -----------------------------------------------------------------------------
r"""
## Метод Рунге-Кутты 2 порядка

### Физическое описание системы

Для затухающего гармонического осциллятора мы имеем систему дифференциальных уравнений:

$$\frac{d\vec{Y}}{dt} = \vec{F}(\vec{Y})$$

где вектор состояния:
$$\vec{Y} = \begin{pmatrix}x \\ v \end{pmatrix}$$

а векторная функция:
$$\vec{F}(\vec{Y}) = \begin{pmatrix} v \\ -\frac{k}{m} x - \frac{b}{m} v \end{pmatrix}$$

### Метод средней точки (RK2)

Метод Рунге-Кутты второго порядка использует две оценки производной:

1. **Первая оценка (в точке n):**
   $$\vec{k}_1 = \vec{F}(\vec{Y}_n)$$

2. **Вторая оценка (в средней точке):**
   $$\vec{k}_2 = \vec{F}\left(\vec{Y}_n + \vec{k}_1 \cdot \frac{\Delta t}{2}\right)$$

3. **Итоговое обновление:**
   $$\vec{Y}_{n+1} = \vec{Y}_n + \frac{\vec{k}_1 + \vec{k}_2}{2} \cdot \Delta t$$

### В компонентной форме:

$$\vec{k}_1 = \begin{pmatrix} v_n \\ -\frac{k}{m} x_n - \frac{b}{m} v_n \end{pmatrix}$$

$$\vec{k}_2 = \begin{pmatrix} v_n + k_{1,v} \cdot \frac{\Delta t}{2} \\ -\frac{k}{m} (x_n + k_{1,x} \cdot \frac{\Delta t}{2}) - \frac{b}{m} (v_n + k_{1,v} \cdot \frac{\Delta t}{2}) \end{pmatrix}$$

$$\vec{Y}_{n+1} = \vec{Y}_n + \frac{\vec{k}_1 + \vec{k}_2}{2} \cdot \Delta t$$"""

def func_f(y):
    """ Функция изменения для затухающего гармонического осциллятора, принимает np.array(shape=2)"""
    x, v = y
    return np.array([
        v,
        -(k / m) * x - (b / m) * v,
    ])


def rk2_method(x0, v0, dt, n_steps):
    """
    Метод Рунге-Кутты 2 порядка для гармонического осциллятора

    Args:
        x0: начальное смещение, м
        v0: начальная скорость, м/с
        dt: шаг интегрирования, с
        n_steps: число шагов

    Returns:
        x: массив смещений
        v: массив скоростей
    """
    y_state = np.zeros((n_steps, 2))

    y_state[0] = [x0, v0]

    for i in range(n_steps - 1):
        k1 = func_f(y_state[i])
        k2 = func_f(y_state[i] + k1 * dt)
        y_state[i+1] = y_state[i] + (k1 + k2) * dt / 2

    # Извлекаем смещение и скорость из состояния
    x = y_state[:, 0]
    v = y_state[:, 1]
    return x, v


def rk4_method(x0, v0, dt, n_steps):
    """
    Метод Рунге-Кутты 4 порядка для гармонического осциллятора

    Args:
        x0: начальное смещение, м
        v0: начальная скорость, м/с
        dt: шаг интегрирования, с
        n_steps: число шагов

    Returns:
        x: массив смещений
        v: массив скоростей
    """
    y_state = np.zeros((n_steps, 2))

    y_state[0] = [x0, v0]

    for i in range(n_steps - 1):
        k1 = func_f(y_state[i])
        k2 = func_f(y_state[i] + k1 * dt/2)
        k3 = func_f(y_state[i] + k2 * dt/2)
        k4 = func_f(y_state[i] + k3 * dt)

        y_state[i+1] = y_state[i] + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

    # Извлекаем смещение и скорость из состояния
    x = y_state[:, 0]
    v = y_state[:, 1]
    return x, v

def adams2_method(x0, v0, dt, n_steps):
    """
    Метод Адамса-Бэшфорта 2-го порядка для гармонического осциллятора

    Args:
        x0: начальное смещение, м
        v0: начальная скорость, м/с
        dt: шаг интегрирования, с
        n_steps: число шагов

    Returns:
        x: массив смещений
        v: массив скоростей
    """
    y_state = np.zeros((n_steps, 2))
    f_history = np.zeros((n_steps, 2))  # история производных

    y_state[0] = [x0, v0]
    f_history[0] = func_f(y_state[0])

    # Для старта метода Адамса нужен 1 предыдущий шаг
    # Используем RK2 вместо Эйлера для лучшего старта (Адамс-Бэшфорта часто неустойчив)
    if n_steps > 1:
        k1 = func_f(y_state[0])
        k2 = func_f(y_state[0] + dt * k1)
        y_state[1] = y_state[0] + (dt / 2) * (k1 + k2)
        f_history[1] = func_f(y_state[1])

    # Применяем метод Адамса-Бэшфорта 2-го порядка для оставшихся шагов
    for i in range(1, n_steps - 1):
        # Формула Адамса-Бэшфорта 2-го порядка:
        # y_{n+1} = y_n + (h/2) * (3*f_n - f_{n-1})
        y_state[i+1] = y_state[i] + (dt / 2) * (
            3 * f_history[i] -
            1 * f_history[i-1]
        )
        f_history[i+1] = func_f(y_state[i+1])

    # Извлекаем смещение и скорость из состояния
    x = y_state[:, 0]
    v = y_state[:, 1]
    return x, v

def adams4_method(x0, v0, dt, n_steps):
    """
    Метод Адамса-Бэшфорта 4-го порядка для гармонического осциллятора

    Args:
        x0: начальное смещение, м
        v0: начальная скорость, м/с
        dt: шаг интегрирования, с
        n_steps: число шагов

    Returns:
        x: массив смещений
        v: массив скоростей
    """
    y_state = np.zeros((n_steps, 2))
    f_history = np.zeros((n_steps, 2))  # история производных

    y_state[0] = [x0, v0]
    f_history[0] = func_f(y_state[0])

    # Для старта метода Адамса нужны 3 предыдущие точки
    # Используем RK4 для первых 3 шагов
    for i in range(min(3, n_steps - 1)):
        k1 = func_f(y_state[i])
        k2 = func_f(y_state[i] + k1 * dt/2)
        k3 = func_f(y_state[i] + k2 * dt/2)
        k4 = func_f(y_state[i] + k3 * dt)

        y_state[i+1] = y_state[i] + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        f_history[i+1] = func_f(y_state[i+1])

    # Применяем метод Адамса-Бэшфорта 4-го порядка для оставшихся шагов
    for i in range(3, n_steps - 1):
        # Формула Адамса-Бэшфорта 4-го порядка:
        # y_{n+1} = y_n + (h/24) * (55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
        y_state[i+1] = y_state[i] + (dt / 24) * (
            55 * f_history[i] -
            59 * f_history[i-1] +
            37 * f_history[i-2] -
            9 * f_history[i-3]
        )
        f_history[i+1] = func_f(y_state[i+1])

    # Извлекаем смещение и скорость из состояния
    x = y_state[:, 0]
    v = y_state[:, 1]
    return x, v

def adams_moulton2_method(x0, v0, dt, n_steps):
    """
    Неявный метод Адамса-Моултона 2-го порядка для гармонического осциллятора
    (метод трапеций)

    Args:
        x0: начальное смещение, м
        v0: начальная скорость, м/с
        dt: шаг интегрирования, с
        n_steps: число шагов

    Returns:
        x: массив смещений
        v: массив скоростей
    """
    from scipy.optimize import fsolve

    y_state = np.zeros((n_steps, 2))
    f_history = np.zeros((n_steps, 2))  # история производных

    y_state[0] = [x0, v0]
    f_history[0] = func_f(y_state[0])

    # Применяем неявный метод Адамса-Моултона 2-го порядка с самого начала
    for i in range(n_steps - 1):
        if i == 0:
            # Для первого шага используем явный метод Эйлера как предиктор
            y_predict = y_state[i] + dt * f_history[i]
        else:
            # Для последующих шагов используем линейную экстраполяцию
            # y_predict = y_n + h * (3/2 * f_n - 1/2 * f_{n-1})
            y_predict = y_state[i] + dt * (1.5 * f_history[i] - 0.5 * f_history[i-1])

        # Корректор: решаем нелинейное уравнение метода трапеций
        # y_{n+1} = y_n + (h/2) * (f_n + f_{n+1})
        def implicit_eq(y_next):
            f_next = func_f(y_next)
            return y_next - y_state[i] - (dt / 2) * (f_history[i] + f_next)

        # Решаем нелинейное уравнение
        y_state[i+1] = fsolve(implicit_eq, y_predict)
        f_history[i+1] = func_f(y_state[i+1])

    # Извлекаем смещение и скорость из состояния
    x = y_state[:, 0]
    v = y_state[:, 1]
    return x, v

def adams_moulton4_method(x0, v0, dt, n_steps):
    """
    Неявный метод Адамса-Моултона 4-го порядка для гармонического осциллятора

    Args:
        x0: начальное смещение, м
        v0: начальная скорость, м/с
        dt: шаг интегрирования, с
        n_steps: число шагов

    Returns:
        x: массив смещений
        v: массив скоростей
    """
    from scipy.optimize import fsolve

    y_state = np.zeros((n_steps, 2))
    f_history = np.zeros((n_steps, 2))  # история производных

    y_state[0] = [x0, v0]
    f_history[0] = func_f(y_state[0])

    # Для старта метода Адамса нужны 3 предыдущие точки
    # Используем RK4 для первых 3 шагов
    for i in range(min(3, n_steps - 1)):
        k1 = func_f(y_state[i])
        k2 = func_f(y_state[i] + k1 * dt/2)
        k3 = func_f(y_state[i] + k2 * dt/2)
        k4 = func_f(y_state[i] + k3 * dt)

        y_state[i+1] = y_state[i] + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        f_history[i+1] = func_f(y_state[i+1])

    # Применяем неявный метод Адамса-Моултона 4-го порядка
    for i in range(3, n_steps - 1):
        # Предиктор: используем Адамса-Бэшфорта для оценки
        y_predict = y_state[i] + (dt / 24) * (
            55 * f_history[i] -
            59 * f_history[i-1] +
            37 * f_history[i-2] -
            9 * f_history[i-3]
        )

        # Корректор: решаем нелинейное уравнение
        # y_{n+1} = y_n + (h/24) * (9*f_{n+1} + 19*f_n - 5*f_{n-1} + f_{n-2})
        def implicit_eq(y_next):
            f_next = func_f(y_next)
            return y_next - y_state[i] - (dt / 24) * (
                9 * f_next +
                19 * f_history[i] -
                5 * f_history[i-1] +
                1 * f_history[i-2]
            )

        # Решаем нелинейное уравнение
        y_state[i+1] = fsolve(implicit_eq, y_predict)
        f_history[i+1] = func_f(y_state[i+1])

    # Извлекаем смещение и скорость из состояния
    x = y_state[:, 0]
    v = y_state[:, 1]
    return x, v    

# -----------------------------------------------------------------------------
# Метод Leapfrog
# -----------------------------------------------------------------------------
r"""
## Метод Leapfrog (симплектический метод чередования сетки)

### Концепция двух чередующихся состояний системы

Метод Leapfrog основан на концепции **чередования сетки** с состояниями на целом и полуцелом шагах:

**Состояния на целом шаге (n, n+1, ...):**
- Смещение x_n
- Ускорение a_n = -(k/m) x_n

**Состояния на полуцелом шаге (n+1/2, n+3/2, ...):**
- Скорость v_{n+1/2}

### Расширенное векторное представление

**Вектор положения (на целом шаге):**
$$\vec{Y}_n = \begin{pmatrix} x_n \end{pmatrix}$$

**Вектор скорости (на полуцелом шаге):**
$$\vec{V}_{n+1/2} = \begin{pmatrix} v_{n+1/2} \end{pmatrix}$$

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
    """Функция для обновления положения (принимает скорость)"""
    return np.array([v])

def func_v(u, v_current=None):
    """
    Функция для обновления скорости (принимает положение и опционально текущую скорость для трения)
    Для диссипативных систем используем приближение с трением
    """
    x = u[0]
    if v_current is None:
        # Без трения (консервативный случай)
        return np.array([-(k / m) * x])
    else:
        # С трением (диссипативный случай)
        return np.array([-(k / m) * x - (b / m) * v_current])

def verlet2_method(x0, v0, dt, n_steps):
    """
    Метод Leapfrog 2-го порядка для гармонического осциллятора
    (Velocity Verlet - эквивалентная форма)

    ДВА ВЫЧИСЛЕНИЯ УСКОРЕНИЯ ЗА ЦИКЛ (как в истинном Leapfrog 2-го порядка):
    1. a_current = a(x_n, v_half) - ускорение в текущей точке
    2. a_next = a(x_{n+1}, v_{n+1/2}) - ускорение в следующей точке

    Это необходимо для достижения 2-го порядка точности.
    Для сравнения: одно вычисление дает порядок O(h), два вычисления дают O(h²).

    Args:
        x0: начальное смещение, м
        v0: начальная скорость, м/с
        dt: шаг интегрирования, с
        n_steps: число шагов

    Returns:
        x: массив смещений
        v: массив скоростей
    """

    u_state = np.zeros((n_steps, 1))  # положение [x]
    v_state = np.zeros((n_steps, 1))  # скорость [v]

    u_state[0] = [x0]
    v_state[0] = [v0]

    # Первое вычисление ускорения
    a_current = func_v(u_state[0], v_state[0][0])[0]

    for i in range(n_steps - 1):
        # Velocity Verlet схема (эквивалент Leapfrog):
        # 1. Обновляем положение: x_{n+1} = x_n + dt*v_n + (dt²/2)*a_n
        u_state[i+1] = u_state[i] + v_state[i] * dt + 0.5 * a_current * dt**2

        # 2. Вычисляем новое ускорение: a_{n+1} = a(x_{n+1}, v_{n+1/2})
        v_half = v_state[i] + 0.5 * a_current * dt  # скорость в середине шага
        a_next = func_v(u_state[i+1], v_half[0])[0]

        # 3. Обновляем скорость: v_{n+1} = v_n + (dt/2)*(a_n + a_{n+1})
        v_state[i+1] = v_state[i] + 0.5 * (a_current + a_next) * dt

        # Сохраняем ускорение для следующего шага
        a_current = a_next

    x = u_state[:, 0]
    v = v_state[:, 0]
    return x, v

def leapfrog_method(x0, v0, dt, n_steps):
    """
    Модифицированный Leapfrog с ОДНИМ вычислением ускорения за цикл
    (приближение 1-го порядка, но очень эффективное)

    ОДНО ВЫЧИСЛЕНИЕ УСКОРЕНИЯ ЗА ЦИКЛ:
    - Ускорение вычисляется только в конце положения
    - Это дает порядок O(h), но требует меньше вычислений
    - Полезно для очень эффективных симуляций

    Args:
        x0: начальное смещение, м
        v0: начальная скорость, м/с
        dt: шаг интегрирования, с
        n_steps: число шагов

    Returns:
        x: массив смещений
        v: массив скоростей
    """

    u_state = np.zeros((n_steps, 1))  # положение [x]
    v_state = np.zeros((n_steps, 1))  # скорость [v]
    v_half_state = np.zeros((n_steps, 1)) # скорость на полуцелом шаге

    u_state[0] = [x0]
    v_state[0] = [v0]

    v_half_state[0] = v_state[0] + func_v(u_state[0], v_state[0][0]) * dt / 2

    for i in range(n_steps - 1):
        # Шаг для положения
        u_state[i+1] = u_state[i] + func_u(v_half_state[i][0]) * dt
        # Шаг для скорости (с учетом трения на текущей скорости)
        v_half_state[i+1] = v_half_state[i] + func_v(u_state[i+1], v_half_state[i][0]) * dt

    v_state[1:] = (v_half_state[:-1] + v_half_state[1:])/2

    x = u_state[:, 0]
    v = v_state[:, 0]
    return x, v


# =============================================================================
# Решение задачами всеми методами
# =============================================================================

# Аналитическое решение
x_analytic, v_analytic = analytical_solution(x0, v0, dt, n_steps)
energy_analytic = energy(x_analytic, v_analytic)

print(f"Начальная энергия (аналитическое решение): {energy_analytic[0]:.6f} Дж")
print(f"Конечная энергия (аналитическое решение): {energy_analytic[-1]:.6f} Дж")
print(f"Изменение энергии: {energy_analytic[-1] - energy_analytic[0]:.6f} Дж")
print(f"Коэффициент затухания: {b/(2*m):.3f} 1/с")

omega0 = np.sqrt(k/m)
gamma = b/(2*m)
if omega0 > gamma:
    print(f"Частота затухающих колебаний: {np.sqrt(omega0**2 - gamma**2):.3f} рад/с")
    print(f"Период затухающих колебаний: {2*pi/np.sqrt(omega0**2 - gamma**2):.3f} с")
else:
    print("Система находится в апериодическом режиме (критическое или перекритическое затухание)")

# Решение методом Leapfrog (2-го порядка, два вычисления ускорения)
x_verlet2, v_verlet2 = verlet2_method(x0, v0, dt, n_steps)
energy_verlet2 = energy(x_verlet2, v_verlet2)

# Решение методом Leapfrog с одним вычислением ускорения (эффективный вариант)
x_leapfrog, v_leapfrog = leapfrog_method(x0, v0, dt, n_steps)
energy_leapfrog = energy(x_leapfrog, v_leapfrog)

# Решение методом Рунге-Кутты 2 порядка
x_rk2, v_rk2 = rk2_method(x0, v0, dt, n_steps)
energy_rk2 = energy(x_rk2, v_rk2)

# Решение методом Рунге-Кутты 4 порядка
x_rk4, v_rk4 = rk4_method(x0, v0, dt, n_steps)
energy_rk4 = energy(x_rk4, v_rk4)

# Решение методом Адамса 2-го порядка
x_adams2, v_adams2 = adams2_method(x0, v0, dt, n_steps)
energy_adams2 = energy(x_adams2, v_adams2)

# Решение методом Адамса 4-го порядка
x_adams4, v_adams4 = adams4_method(x0, v0, dt, n_steps)
energy_adams4 = energy(x_adams4, v_adams4)

# Решение методом Адамса-Моултона 2-го порядка (неявный)
x_adams_moulton2, v_adams_moulton2 = adams_moulton2_method(x0, v0, dt, n_steps)
energy_adams_moulton2 = energy(x_adams_moulton2, v_adams_moulton2)

# Решение методом Адамса-Моултона 4-го порядка (неявный)
x_adams_moulton4, v_adams_moulton4 = adams_moulton4_method(x0, v0, dt, n_steps)
energy_adams_moulton4 = energy(x_adams_moulton4, v_adams_moulton4)


# Подготавливаем данные для графиков
plot_data = [
    {
        'label': 'verlet2',
        'color': 'blue',
        'x': x_verlet2,
        'v': v_verlet2,
        'e': energy_verlet2
    },
    {
        'label': 'Leapfrog',
        'color': 'cyan',
        'x': x_leapfrog,
        'v': v_leapfrog,
        'e': energy_leapfrog
    },
    # {
    #     'label': 'Метод RK4',
    #     'color': 'orange',
    #     'x': x_rk4,
    #     'v': v_rk4,
    #     'e': energy_rk4
    # },
    {
        'label': 'RK2',
        'color': 'orange',
        'x': x_rk2,
        'v': v_rk2,
        'e': energy_rk2
    },    
    # {
    #     'label': 'Адамс-2',
    #     'color': 'purple',
    #     'x': x_adams2,
    #     'v': v_adams2,
    #     'e': energy_adams2
    # },
    {
        'label': 'Метод Адамса-4',
        'color': 'brown',
        'x': x_adams4,
        'v': v_adams4,
        'e': energy_adams4
    },
    {
        'label': 'Адамса-Моултона-2',
        'color': 'magenta',
        'x': x_adams_moulton2,
        'v': v_adams_moulton2,
        'e': energy_adams_moulton2
    },
    # {
    #     'label': 'Метод Адамса-Моултона-4',
    #     'color': 'cyan',
    #     'x': x_adams_moulton4,
    #     'v': v_adams_moulton4,
    #     'e': energy_adams_moulton4
    # },
    {
        'label': 'Аналитическое',
        'color': 'green',
        'x': x_analytic,
        'v': v_analytic,
        'e': energy_analytic
    }
]



def create_comparison_plots(t, data_list, title="Сравнение методов"):
    """
    Создает графики сравнения методов интегрирования для осциллятора

    Args:
        t: массив времени
        data_list: список словарей с данными методов
                   каждый словарь содержит: 'label', 'color', 'x', 'v', 'e'
        title: заголовок фигуры

    Returns:
        fig, axes: фигура и массив осей matplotlib
    """

    for data in data_list:
        print(f"\nНачальная энергия (метод {data['label']}): {data['e'][0]:.6f} Дж")
        print(f"Конечная энергия (метод {data['label']}): {data['e'][-1]:.6f} Дж")
        print(f"Изменение энергии: {data['e'][-1] - data['e'][0]:.6f} Дж")
        print(f"Относительное изменение: {((data['e'][-1] - data['e'][0])/data['e'][0]*100):.3f}%")


    # Создаем фигуру с подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)

    # 1. Смещение как функция времени
    for data in data_list:
        linestyle = '--' if 'Аналитическое' in data['label'] else '-'
        axes[0, 0].plot(t, data['x'], label=data['label'], linewidth=2,
                       color=data['color'], linestyle=linestyle)
    axes[0, 0].set_xlabel('Время, с')
    axes[0, 0].set_ylabel('Смещение, м')
    axes[0, 0].set_title('Смещение осциллятора')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Фазовый портрет
    for data in data_list:
        linestyle = '--' if 'Аналитическое' in data['label'] else '-'
        axes[0, 1].plot(data['x'], data['v'], label=data['label'], linewidth=2,
                       color=data['color'], linestyle=linestyle)
    axes[0, 1].set_xlabel('Смещение, м')
    axes[0, 1].set_ylabel('Скорость, м/с')
    axes[0, 1].set_title('Фазовый портрет')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Энергия
    for data in data_list:
        linestyle = '--' if 'Аналитическое' in data['label'] else '-'
        axes[1, 0].plot(t, data['e'], label=data['label'], linewidth=2,
                       color=data['color'], linestyle=linestyle)
    axes[1, 0].set_xlabel('Время, с')
    axes[1, 0].set_ylabel('Энергия, Дж')
    axes[1, 0].set_title('Полная энергия')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Пустой subplot для анимации
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Анимация осциллятора')

    plt.tight_layout()
    return fig, axes

# =============================================================================
# Сравнение методов - графики
# =============================================================================

r"""
## Сравнение методов

Теперь построим графики для сравнения методов второго порядка.
"""



# Создаем графики
fig, axes = create_comparison_plots(t, plot_data, 'Сравнение методов второго порядка для затухающего осциллятора')

# 4. Анимация пружинного осциллятора для всех трех методов в четвертом subplot

# Настройка четвертого subplot для анимации
ax_anim = axes[1, 1]
# Устанавливаем пределы - пружина будет от -1.5 до 1.5, с запасом
max_displacement = 1.5
ax_anim.set_xlim(-max_displacement, max_displacement)
ax_anim.set_ylim(-0.5, 0.5)
ax_anim.set_aspect('equal')
ax_anim.grid(True, alpha=0.3)
ax_anim.set_title('Анимация пружинных осцилляторов')

# Фиксированная стенка слева
ax_anim.plot([-max_displacement, -max_displacement], [-0.2, 0.2], 'k-', lw=4)
ax_anim.text(-max_displacement, 0.3, 'Стенка', ha='center', va='bottom')

# Линии для каждого осциллятора (пружины + масса)
spring_lines = []
mass_lines = []

# Используем цвета из plot_data
for data in plot_data:
    # Пружина (будет состоять из нескольких сегментов)
    spring_line, = ax_anim.plot([], [], '-', lw=2, color=data['color'], label=data['label'])
    # Масса (квадратик на конце пружины)
    mass_line, = ax_anim.plot([], [], 's-', markersize=12, color=data['color'], markerfacecolor=data['color'])
    spring_lines.append(spring_line)
    mass_lines.append(mass_line)

ax_anim.legend()

# Текст для энергии
energy_text = ax_anim.text(0.02, 0.98, '', transform=ax_anim.transAxes,
                          verticalalignment='top', fontsize=8,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def animate_pendulums(frame):
    """Функция анимации для обновления пружинных осцилляторов в subplot"""
    energy_info = []
    methods_data = [(data['x'], data['v'], data['label']) for data in plot_data]

    for i, (x, v, label) in enumerate(methods_data):
        # Позиция массы
        mass_x = x[frame]
        mass_y = 0  # масса движется горизонтально

        # Создаем пружину как серию зигзагов
        wall_x = -max_displacement  # позиция стенки
        spring_length = mass_x - wall_x
        n_coils = 8  # количество витков пружины

        # Создаем точки пружины
        spring_points_x = []
        spring_points_y = []

        for j in range(n_coils * 2 + 1):
            # Распределяем точки вдоль пружины
            fraction = j / (n_coils * 2)
            x_pos = wall_x + fraction * spring_length

            # Зигзаг пружины
            if j % 2 == 0:
                y_pos = 0  # центральная линия
            else:
                y_pos = 0.1 * (-1)**(j//2)  # зигзаг вверх-вниз

            spring_points_x.append(x_pos)
            spring_points_y.append(y_pos)

        # Добавляем точку массы
        spring_points_x.append(mass_x)
        spring_points_y.append(mass_y)

        # Обновляем пружину
        spring_lines[i].set_data(spring_points_x, spring_points_y)

        # Обновляем массу
        mass_lines[i].set_data([mass_x], [mass_y])

        # Информация об энергии
        current_energy = energy(x[frame], v[frame])
        energy_info.append('.4f'f'{label}: E = {current_energy:.4f} Дж')

    energy_text.set_text('\n'.join(energy_info))

    return spring_lines + mass_lines + [energy_text]

def init_animation():
    """Инициализация анимации"""
    for spring_line in spring_lines:
        spring_line.set_data([], [])
    for mass_line in mass_lines:
        mass_line.set_data([], [])
    energy_text.set_text('')
    return spring_lines + mass_lines + [energy_text]

# Создаем анимацию в subplot
print("\nСоздание анимации в subplot...")

# Создание анимации для subplot
fps = 20
interval = 1000 / fps  # интервал в миллисекундах
min_length = min(len(x_verlet2), len(x_rk2), len(x_analytic))
frames = np.arange(0, min_length, max(1, min_length//(fps*T)))  # выбираем кадры для плавности

anim = animation.FuncAnimation(
    fig, animate_pendulums, init_func=init_animation,
    frames=len(frames), interval=interval, blit=True
)

plt.tight_layout()
plt.show()

print("Анимация встроена в график! Для отображения в ноутбуке используйте:")
print("HTML(anim.to_jshtml())")

# =============================================================================
# Анимация движения маятника
# =============================================================================

r"""
## Анимация движения маятника

Создадим анимации для демонстрации работы всех трех методов.
"""

# =============================================================================
# Выводы и заключение
# =============================================================================

r"""
## Выводы

На основе проведенного сравнения методов интегрирования для затухающего гармонического осциллятора можно сделать следующие выводы:

### По точности решения:
1. **Методы Рунге-Кутты (RK4)** - наиболее точный одношаговый метод
   - Отличная точность для широкого диапазона задач
   - Стабильный для различных типов систем

2. **Методы Адамса-Моултона (неявные)** - самые точные для жестких систем
   - Высокая точность и устойчивость
   - Требуют решения нелинейных уравнений на каждом шаге
   - 2-го порядка проще в реализации, но менее точный

3. **Методы Адамса-Бэшфорта (явные)** - эффективные многошаговые методы
   - Высокая точность при больших шагах интегрирования
   - Могут быть неустойчивыми для жестких систем

4. **Метод Leapfrog** - хорошая точность для консервативных систем
   - Симплектический, сохраняет структуру фазового пространства
   - Может иметь небольшие отклонения при сильном затухании

### Классификация методов интегрирования:

#### **Явные методы:**
- **Определение**: Новое значение решения вычисляется только через предыдущие значения
- **Примеры**: RK4, Адамса-Бэшфорта, Leapfrog
- **Преимущества**: Простота реализации, высокая скорость
- **Недостатки**: Могут быть неустойчивыми при больших шагах

#### **Неявные методы:**
- **Определение**: Новое значение решения входит в уравнение и требует решения нелинейной системы
- **Примеры**: Адамса-Моултона, методы Розенброка, BDF методы
- **Преимущества**: Устойчивость при больших шагах, хорошая сходимость
- **Недостатки**: Сложность реализации, требуют решения уравнений на каждом шаге

#### **Одношаговые vs Многошаговые:**
- **Одношаговые** (RK4): используют только текущую точку, самодостаточны
- **Многошаговые** (Адамса): используют историю нескольких предыдущих шагов
- **Преимущество многошаговых**: выше эффективность при фиксированной точности

### Рекомендации по применению:
- **Для высокой точности на коротких интервалах**: RK4
- **Для длинных расчетов с умеренной точностью**: Leapfrog
- **Для эффективных расчетов**: Адамса-Бэшфорта (явные)
- **Для жестких систем**: Адамса-Моултона (неявные)
- **Для простых задач**: Адамса-2

**Примечание**: Затухающий осциллятор демонстрирует три режима в зависимости от параметров:
критическое затухание, недозатухающие и перезагужающие колебания.
"""
