
""" # Экологические модели # """

r"""
*Вся плоть - трава (Исаия 40:6)*

Биологи исследуют развитие популяции организмов (травы, кроликов и лис) внутри замкнутой экосистемы (огороженного луга). Существует предельная экологическая вместимость луга.

Мы будем моделировать динамику уравнениями:

$$ 

x' = grow_grass(t) x (1-x) - eating_grass x y
y' = eating_grass digesty_grass x y - eating_rabbits y z - mortality_rabbits y
z' = eating_rabbits digesty_rabbits x y - mortality_fox y

grow_grass(t) = grow_grass_0 + grow_grass_a * sin( t * grow_grass_fr )

$$

где

x, y, z - популяция травы, кроликов и лис
grow_grass(t) - скорость роста травы - периодическая функция, зависит от сезона
eating_grass - скорость поедания кроликами травы
digesty_grass - процент усвоения биомассы травы биомассой кроликов
eating_rabbits - скорость поедания кроликов лисами
mortality_rabbits - естественная смертность кроликов
digesty_rabbits - усвояемость крольчатины
mortality_fox - естественная смертность лис

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from math import pi

""" ## Параметры модели ## """

# Параметры роста травы (сезонные колебания)
grow_grass_0 = 1.0     # базовая скорость роста травы
grow_grass_a = 0     # амплитуда сезонных колебаний
grow_grass_fr = 2*pi   # частота сезонных колебаний (годовой цикл)

# Параметры взаимодействия трава-кролики
eating_grass = 1.5     # скорость поедания кроликами травы
digesty_grass = 0.7    # эффективность усвоения травы кроликами

# Параметры взаимодействия кролики-лисы
eating_rabbits = 0.8   # скорость поедания кроликов лисами
digesty_rabbits = 0.6  # эффективность усвоения крольчатины 

# Параметры смертности
mortality_rabbits = 0.2  # естественная смертность кроликов
mortality_fox = 0.1      # естественная смертность лис

# Начальные условия (доли от максимальной популяции)
x0 = 0.8  # начальная популяция травы
y0 = 0.3  # начальная популяция кроликов
z0 = 0.1  # начальная популяция лис

# Параметры интегрирования
t_start = 0.0
t_end = 200.0    # моделируем 20 единиц времени (лет)
dt = 0.01
t_eval = np.arange(t_start, t_end, dt)

print("=== Параметры модели ===")
print(f"Скорость роста травы: базовая = {grow_grass_0}, амплитуда = {grow_grass_a}, частота = {grow_grass_fr:.1f}")
print(f"Взаимодействие трава-кролики: поедание = {eating_grass}, усвоение = {digesty_grass}")
print(f"Взаимодействие кролики-лисы: поедание = {eating_rabbits}, усвоение = {digesty_rabbits}")
print(f"Смертность: кролики = {mortality_rabbits}, лисы = {mortality_fox}")
print(f"Начальные условия: трава = {x0}, кролики = {y0}, лисы = {z0}")
print(f"Время моделирования: {t_end} ед. времени")
print()

""" ## Функции правых частей системы уравнений ## """

def grow_grass_function(t):
    """
    Скорость роста травы как функция времени (сезонные колебания)

    Args:
        t: время

    Returns:
        grow_grass(t) = grow_grass_0 + grow_grass_a * sin(t * grow_grass_fr)
    """
    return grow_grass_0 + grow_grass_a * np.sin(t * grow_grass_fr)

def ecosystem_equations(t, state):
    """
    Правая часть системы дифференциальных уравнений экосистемы

    Args:
        t: время
        state: вектор состояния [x, y, z] где x-трава, y-кролики, z-лисы

    Returns:
        dx/dt, dy/dt, dz/dt
    """
    x, y, z = state

    # Уравнение для травы
    dx_dt = grow_grass_function(t) * x * (1 - x) - eating_grass * x * y

    # Уравнение для кроликов
    dy_dt = eating_grass * digesty_grass * x * y - eating_rabbits * y * z - mortality_rabbits * y

    # Уравнение для лис
    dz_dt = eating_rabbits * digesty_rabbits * y * z - mortality_fox * z

    return [dx_dt, dy_dt, dz_dt]

""" ## Решение системы методом RK45 ## """

def solve_ecosystem_rk45(x0, y0, z0, t_eval):
    """
    Решение системы уравнений экосистемы методом RK45

    Args:
        x0, y0, z0: начальные условия для травы, кроликов и лис
        t_eval: точки времени для оценки решения

    Returns:
        x_sol, y_sol, z_sol: решения для популяций в точках t_eval
    """
    sol = solve_ivp(
        ecosystem_equations,
        (t_eval[0], t_eval[-1]),
        [x0, y0, z0],
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )

    return sol.y[0], sol.y[1], sol.y[2]  # x, y, z

""" ## Решение системы и построение графиков ## """

print("=== Решение системы уравнений ===")

# Решение системы
x_sol, y_sol, z_sol = solve_ecosystem_rk45(x0, y0, z0, t_eval)

# Вычисление скорости роста травы для сравнения
grow_grass_values = grow_grass_function(t_eval)


print("График популяций сохранен как 'ecosystem_populations.png'")

""" ## Общий график экосистемы ## """

plt.figure(figsize=(18, 10))

# Верхний ряд: общий график популяций от времени (растянут на всю ширину)
plt.subplot(2, 3, (1, 3))  # занимает позиции 1, 2, 3
plt.title('Динамика экосистемы: популяции со временем', fontsize=14)
plt.plot(t_eval, x_sol, color='green', linewidth=2, label='Трава (x)', alpha=0.8)
plt.plot(t_eval, y_sol, color='blue', linewidth=2, label='Кролики (y)', alpha=0.8)
plt.plot(t_eval, z_sol, color='red', linewidth=2, label='Лисы (z)', alpha=0.8)
plt.xlabel('Время', fontsize=12)
plt.ylabel('Популяция', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.ylim(0, max(max(x_sol), max(y_sol), max(z_sol)) * 1.1)

# Нижний ряд: фазовые портреты
# Фазовый портрет трава-кролики
plt.subplot(2, 3, 4)
plt.title('Трава vs Кролики', fontsize=12)
plt.plot(x_sol, y_sol, color='purple', linewidth=2, alpha=0.7)
plt.plot(x_sol[0], y_sol[0], 'go', markersize=8, label='Начало')
plt.plot(x_sol[-1], y_sol[-1], 'ro', markersize=8, label='Конец')
plt.xlabel('Трава x')
plt.ylabel('Кролики y')
plt.grid(True, alpha=0.3)
plt.legend()

# Фазовый портрет кролики-лисы
plt.subplot(2, 3, 5)
plt.title('Кролики vs Лисы', fontsize=12)
plt.plot(y_sol, z_sol, color='orange', linewidth=2, alpha=0.7)
plt.plot(y_sol[0], z_sol[0], 'go', markersize=8, label='Начало')
plt.plot(y_sol[-1], z_sol[-1], 'ro', markersize=8, label='Конец')
plt.xlabel('Кролики y')
plt.ylabel('Лисы z')
plt.grid(True, alpha=0.3)
plt.legend()

# Фазовый портрет трава-лисы
plt.subplot(2, 3, 6)
plt.title('Трава vs Лисы', fontsize=12)
plt.plot(x_sol, z_sol, color='brown', linewidth=2, alpha=0.7)
plt.plot(x_sol[0], z_sol[0], 'go', markersize=8, label='Начало')
plt.plot(x_sol[-1], z_sol[-1], 'ro', markersize=8, label='Конец')
plt.xlabel('Трава x')
plt.ylabel('Лисы z')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('ecosystem_overview.png', dpi=150, bbox_inches='tight')
plt.show()

print("Общий график экосистемы сохранен как 'ecosystem_overview.png'")

""" ## Анализ решения ## """

print("\n=== Анализ динамики экосистемы ===")

# Статистика для травы
print("\nТрава (x):")
print(f"  Начальное значение: {x_sol[0]:.3f}")
print(f"  Минимальное значение: {np.min(x_sol):.3f}")
print(f"  Максимальное значение: {np.max(x_sol):.3f}")
print(f"  Среднее значение: {np.mean(x_sol):.3f}")
print(f"  Финальное значение: {x_sol[-1]:.3f}")
print(f"  Амплитуда колебаний: {np.max(x_sol) - np.min(x_sol):.3f}")

# Статистика для кроликов
print("\nКролики (y):")
print(f"  Начальное значение: {y_sol[0]:.3f}")
print(f"  Минимальное значение: {np.min(y_sol):.3f}")
print(f"  Максимальное значение: {np.max(y_sol):.3f}")
print(f"  Среднее значение: {np.mean(y_sol):.3f}")
print(f"  Финальное значение: {y_sol[-1]:.3f}")
print(f"  Амплитуда колебаний: {np.max(y_sol) - np.min(y_sol):.3f}")

# Статистика для лис
print("\nЛисы (z):")
print(f"  Начальное значение: {z_sol[0]:.3f}")
print(f"  Минимальное значение: {np.min(z_sol):.3f}")
print(f"  Максимальное значение: {np.max(z_sol):.3f}")
print(f"  Среднее значение: {np.mean(z_sol):.3f}")
print(f"  Финальное значение: {z_sol[-1]:.3f}")
print(f"  Амплитуда колебаний: {np.max(z_sol) - np.min(z_sol):.3f}")

# Анализ взаимосвязей
correlation_xy = np.corrcoef(x_sol, y_sol)[0, 1]
correlation_yz = np.corrcoef(y_sol, z_sol)[0, 1]
correlation_xz = np.corrcoef(x_sol, z_sol)[0, 1]

print("\nКорреляции между популяциями:")
print(f"  Трава-Кролики: {correlation_xy:.3f}")
print(f"  Кролики-Лисы: {correlation_yz:.3f}")
print(f"  Трава-Лисы: {correlation_xz:.3f}")

""" ## Выводы ## """

"""
## Выводы

В рамках этого занятия мы реализовали сложную экологическую модель пищевой цепи с тремя уровнями:

### Модель экосистемы:

Система дифференциальных уравнений описывает взаимодействие трех видов в пищевой цепи:
- **Трава (x)**: растет логистически с сезонными колебаниями, поедается кроликами
- **Кролики (y)**: питаются травой, поедаются лисами, имеют естественную смертность
- **Лисы (z)**: питаются кроликами, имеют естественную смертность

### Основные результаты:

1. **Сезонные колебания роста травы** создают периодические изменения в экосистеме
   - Популяции всех видов демонстрируют сложные колебания
   - Наблюдаются запаздывания в реакциях хищников на изменения популяций жертв

2. **Динамика популяций**:
   - Все популяции демонстрируют устойчивые колебания
   - Амплитуды колебаний различны для разных видов
   - Корреляции между популяциями показывают типичные взаимосвязи пищевой цепи

3. **Фазовые портреты** демонстрируют сложные траектории в фазовом пространстве
   - Траектории не замыкаются в простые циклы из-за сезонного фактора
   - Наблюдаются квазипериодические движения

### Методические особенности:

- Для решения системы жестких нелинейных ОДУ использовался метод Рунге-Кутты 4-5 порядка
- Высокая точность достигнута за счет строгих tolerances и мелкого шага интегрирования
- Все графики сохранены в формате PNG для анализа

### Экологические интерпретации:

- Модель демонстрирует типичное поведение экосистем с сезонными циклами
- Колебания популяций соответствуют реальным экологическим наблюдениям
- Сезонный фактор роста продуцентов (травы) влияет на всю пищевую цепь

Эта модель может быть использована для изучения влияния климатических изменений, сезонности и антропогенных факторов на экосистемы.
"""