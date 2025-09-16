r"""
# Численные методы решения ОДУ на примере затухающего гармонического осциллятора

На этом занятии мы реализуем и сравним ряд численных методов интегрирования:
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
from scipy.optimize import fsolve

from oscillator_plots import plots

r"""
## Уравнение гармонического осциллятора

$$ m\ddot{u} + b\dot{u} + ku = 0 $$

где:
- $m$ — масса,
- $b$ — коэффициент трения,
- $k$ — жесткость пружины,
- $u$ — смещение.
"""

# Физические параметры гармонического осциллятора
m = 1.0   # масса, кг
k = 1000.0  # жесткость пружины, Н/м
b = 0.0  # коэффициент трения, кг/с

# Начальные условия
u0 = 0.5   # начальное смещение, м
v0 = 0.0   # начальная скорость, м/с

# Параметры интегрирования
T = 10.0          # время моделирования, с (уменьшили для жесткой системы)
dt = 0.01      # шаг интегрирования, с (уменьшили для стабильности Адамса)
n_steps = int(T/dt)   # число шагов

# Массив времени
t = np.arange(0, T, dt)

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

def analytical_method(u0, v0, dt, n_steps):
    """
    Аналитическое решение для затухающего гармонического осциллятора

    Args:
        u0: начальное положение, м
        v0: начальная скорость, м/с
        dt: шаг времени
        n_steps: число шагов

    Returns:
        u_analytical: массив положений
        v_analytical: массив скоростей
    """

    T = n_steps*dt
    # Параметры затухающего осциллятора
    omega = np.sqrt(k / m)  # собственная частота
    gamma = b / (2 * m)      # коэффициент затухания

    # Определяем тип колебаний
    discriminant = omega**2 - gamma**2

    t = np.arange(0, T, dt)

    if discriminant > 0:  # Недозатухающий режим
        omega = np.sqrt(discriminant)
        A = u0
        B = (v0 + gamma * u0) / omega

        u = np.exp(-gamma * t) * (A * np.cos(omega * t) + B * np.sin(omega * t))
        v = np.exp(-gamma * t) * ((-gamma * A - omega * B) * np.cos(omega * t) +
                                  (-gamma * B + omega * A) * np.sin(omega * t))

    elif discriminant < 0:  # Перезагужающий режим
        alpha = np.sqrt(-discriminant)
        A = u0
        B = (v0 + gamma * u0) / alpha

        u = np.exp(-gamma * t) * (A * np.cosh(alpha * t) + B * np.sinh(alpha * t))
        v = np.exp(-gamma * t) * ((-gamma * A - alpha * B) * np.cosh(alpha * t) +
                                  (-gamma * B + alpha * A) * np.sinh(alpha * t))

    else:  # Критическое затухание
        A = u0
        B = v0 + gamma * u0

        u = np.exp(-gamma * t) * (A + B * t)
        v = np.exp(-gamma * t) * (-gamma * A - gamma * B * t + B)

    return u, v

def func_f(y):
    """ Функция изменения для затухающего гармонического осциллятора, принимает np.array(shape=2)"""
    u, v = y
    return np.array([
        v,
        -(k / m) * u - (b / m) * v,
    ])

def rk2_method(u0, v0, dt, n_steps):

    y_state = np.zeros((n_steps, 2))

    y_state[0] = [u0, v0]

    for i in range(n_steps - 1):
        k1 = func_f(y_state[i])
        k2 = func_f(y_state[i] + k1 * dt)
        y_state[i+1] = y_state[i] + (k1 + k2) * dt / 2

    # Извлекаем смещение и скорость из состояния
    u = y_state[:, 0]
    v = y_state[:, 1]
    return u, v

def rk4_method(u0, v0, dt, n_steps):

    y_state = np.zeros((n_steps, 2))

    y_state[0] = [u0, v0]

    for i in range(n_steps - 1):
        k1 = func_f(y_state[i])
        k2 = func_f(y_state[i] + k1 * dt/2)
        k3 = func_f(y_state[i] + k2 * dt/2)
        k4 = func_f(y_state[i] + k3 * dt)

        y_state[i+1] = y_state[i] + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

    # Извлекаем смещение и скорость из состояния
    u = y_state[:, 0]
    v = y_state[:, 1]
    return u, v

def explicit_adams2_method(u0, v0, dt, n_steps):
    u = np.zeros((n_steps, 1))
    v = np.zeros((n_steps, 1))

    # TODO

    return u, v

def explicit_adams4_method(u0, v0, dt, n_steps):

    u = np.zeros((n_steps, 1))
    v = np.zeros((n_steps, 1))

    # TODO

    return u, v


def implicit_adams2_method(u0, v0, dt, n_steps):

    u = np.zeros((n_steps, 1))
    v = np.zeros((n_steps, 1))

    # TODO

    return u, v

def implicit_adams4_method(u0, v0, dt, n_steps):

    u = np.zeros((n_steps, 1))
    v = np.zeros((n_steps, 1))

    # TODO

    return u, v    

def verlet2_method(u0, v0, dt, n_steps):

    u = np.zeros((n_steps, 1))
    v = np.zeros((n_steps, 1))

    # TODO

    return u, v

# =============================================================================
# Решение задачами всеми методами
# =============================================================================

# Аналитическое решение
u_analytic, v_analytic = analytical_method(u0, v0, dt, n_steps)
e_analytic = energy(u_analytic, v_analytic)

print(f"Начальная энергия (аналитическое решение): {e_analytic[0]:.6f} Дж")
print(f"Конечная энергия (аналитическое решение): {e_analytic[-1]:.6f} Дж")
print(f"Изменение энергии: {e_analytic[-1] - e_analytic[0]:.6f} Дж")
print(f"Коэффициент затухания: {b/(2*m):.3f} 1/с")

omega0 = np.sqrt(k/m)
gamma = b/(2*m)
if omega0 > gamma:
    print(f"Частота затухающих колебаний: {np.sqrt(omega0**2 - gamma**2):.3f} рад/с")
    print(f"Период затухающих колебаний: {2*pi/np.sqrt(omega0**2 - gamma**2):.3f} с")
else:
    print("Система находится в апериодическом режиме (критическое или перекритическое затухание)")

# Решение методом Leapfrog (2-го порядка, два вычисления ускорения)
u_verlet2, v_verlet2 = verlet2_method(u0, v0, dt, n_steps)
e_verlet2 = energy(u_verlet2, v_verlet2)

# Решение методом Leapfrog с одним вычислением ускорения (эффективный вариант)
u_verlet2, v_verlet2 = verlet2_method(u0, v0, dt, n_steps)
e_verlet2 = energy(u_verlet2, v_verlet2)

# Решение методом Рунге-Кутты 2 порядка
u_rk2, v_rk2 = rk2_method(u0, v0, dt, n_steps)
e_rk2 = energy(u_rk2, v_rk2)

# Решение методом Рунге-Кутты 4 порядка
u_rk4, v_rk4 = rk4_method(u0, v0, dt, n_steps)
e_rk4 = energy(u_rk4, v_rk4)

# Решение явным методом Адамса (-Бэшфорта) 2-го порядка
u_explicit_adams2, v_explicit_adams2 = explicit_adams2_method(u0, v0, dt, n_steps)
e_explicit_adams2 = energy(u_explicit_adams2, v_explicit_adams2)

# Решение явным методом Адамса (-Бэшфорта) 4-го порядка
u_explicit_adams4, v_explicit_adams4 = explicit_adams4_method(u0, v0, dt, n_steps)
e_explicit_adams4 = energy(u_explicit_adams4, v_explicit_adams4)

# Решение неявным методом Адамса (-Моултона) 2-го порядка (неявный)
u_implicit_adams2, v_implicit_adams2 = implicit_adams2_method(u0, v0, dt, n_steps)
e_implicit_adams2 = energy(u_implicit_adams2, v_implicit_adams2)

# Решение неявным методом Адамса (-Моултона) 4-го порядка (неявный)
u_implicit_adams4, v_implicit_adams4 = implicit_adams4_method(u0, v0, dt, n_steps)
e_implicit_adams4 = energy(u_implicit_adams4, v_implicit_adams4)


# Подготавливаем данные для графиков
plot_data = [
    {
        'label': 'verlet2',
        'color': 'blue',
        'u': u_verlet2,
        'v': v_verlet2,
        'e': e_verlet2
    },
    {
        'label': 'Leapfrog',
        'color': 'cyan',
        'u': u_verlet2,
        'v': v_verlet2,
        'e': e_verlet2
    },
    {
        'label': 'Метод RK4',
        'color': 'orange',
        'u': u_rk4,
        'v': v_rk4,
        'e': e_rk4
    },
    {
        'label': 'RK2',
        'color': 'orange',
        'u': u_rk2,
        'v': v_rk2,
        'e': e_rk2
    },    
    {
        'label': 'Адамс-2',
        'color': 'purple',
        'u': u_explicit_adams2,
        'v': v_explicit_adams2,
        'e': e_explicit_adams2
    },
    {
        'label': 'явный Адамс-4',
        'color': 'brown',
        'u': u_explicit_adams4,
        'v': v_explicit_adams4,
        'e': e_explicit_adams4
    },
    {
        'label': 'неявный Адамс-2',
        'color': 'magenta',
        'u': u_implicit_adams2,
        'v': v_implicit_adams2,
        'e': e_implicit_adams2
    },
    {
        'label': 'неявный Адамс-4',
        'color': 'cyan',
        'u': u_implicit_adams4,
        'v': v_implicit_adams4,
        'e': e_implicit_adams4
    },
    {
        'label': 'Аналитическое',
        'color': 'green',
        'u': u_analytic,
        'v': v_analytic,
        'e': e_analytic
    }
]

plots(t, T, plot_data)
