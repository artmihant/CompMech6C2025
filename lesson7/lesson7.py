""" # Гиперболические уравнения: от переноса до ударных волн # """

r"""
На этом занятии мы исследуем численные методы решения гиперболических уравнений 
на примере линейного уравнения переноса (адвекции):

$$\frac{\partial u}{\partial t} + c \cdot \frac{\partial u}{\partial x} = 0$$

Это уравнение описывает перенос профиля $u(x,t)$ со скоростью $c$ без изменения формы.
Аналитическое решение: $u(x,t) = u(x - ct, 0)$.

Мы реализуем и сравним четыре численные схемы:
1. **FTCS** (Forward Time, Centered Space) - неустойчивая схема
2. **Upwind** - схема первого порядка, стабильная, но диссипативная
3. **Lax-Wendroff** - схема второго порядка с дисперсией
4. **Leap-frog** - схема второго порядка с минимальной диссипацией
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


""" ## Параметры задачи ## """

# Пространственная область
L = 10.0          # длина области, м
Nx = 200          # количество узлов сетки
dx = L / (Nx - 1) # шаг по пространству

# Временные параметры
c = 1.0           # скорость переноса, м/с
T = 8.0           # время симуляции, с
dt = 0.04         # шаг по времени, с (может потребоваться корректировка)
Nt = int(T / dt)  # количество временных шагов

# Число Куранта (CFL number)
sigma = c * dt / dx

# Сетка
x = np.linspace(0, L, Nx)

print(f"Параметры численной схемы:")
print(f"  Шаг по пространству dx = {dx:.4f} м")
print(f"  Шаг по времени dt = {dt:.4f} с")
print(f"  Число Куранта σ = c·dt/dx = {sigma:.4f}")
print(f"  Условие устойчивости CFL: σ ≤ 1.0")
print(f"  Количество временных шагов: {Nt}")


""" ## 1. Начальное условие## """

def initial_condition(x, x_center=3.0, width=1.0, height=0.5):
    """
    Создает начальный профиль
    """
    u = np.zeros_like(x)
    
    # Тело котика - полукруг
    body_mask = np.abs(x - x_center) <= width / 2
    x_body = x[body_mask]
    radius = width / 2
    u[body_mask] = height * np.sqrt(1 - ((x_body - x_center) / radius)**2)
    
    # Левое ухо - треугольник
    left_ear_x = x_center - width * 0.35
    left_ear_width = width * 0.15
    left_ear_mask = np.abs(x - left_ear_x) <= left_ear_width
    x_left_ear = x[left_ear_mask]
    u[left_ear_mask] = np.maximum(u[left_ear_mask], 
                                    height * 1.3 * (1 - np.abs(x_left_ear - left_ear_x) / left_ear_width))
    
    # Правое ухо - треугольник
    right_ear_x = x_center + width * 0.35
    right_ear_width = width * 0.15
    right_ear_mask = np.abs(x - right_ear_x) <= right_ear_width
    x_right_ear = x[right_ear_mask]
    u[right_ear_mask] = np.maximum(u[right_ear_mask], 
                                     height * 1.3 * (1 - np.abs(x_right_ear - right_ear_x) / right_ear_width))
    
    return u


# Создаем и визуализируем начальное условие
u_initial = initial_condition(x)

plt.figure(figsize=(10, 4))
plt.plot(x, u_initial, 'b-', linewidth=2, label='Йа бэтмен')
plt.xlabel('x, м')
plt.ylabel('u')
plt.title('Начальное состояние волны')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


""" ## 2. Схinitial_conditione, Centered Space) ## """

def solve_advection_ftcs(u0, x, c, dt, dx, Nt):
    """
    Решение уравнения переноса методом FTCS
    
    Схема: (u_i^{n+1} - u_i^n) / dt + c * (u_{i+1}^n - u_{i-1}^n) / (2*dx) = 0
    
    ВНИМАНИЕ: Эта схема безусловно неустойчива для линейного уравнения переноса!
    
    Args:
        u0: начальное условие
        x: пространственная сетка
        c: скорость переноса
        dt: шаг по времени
        dx: шаг по пространству
        Nt: количество временных шагов
        
    Returns:
        u_history: история решения (Nt+1, Nx)
        t_history: временная сетка
    """
    Nx = len(x)
    u = u0.copy()
    u_history = [u.copy()]
    t_history = [0.0]
    
    sigma = c * dt / (2 * dx)
    
    for n in range(Nt):
        u_new = u.copy()
        
        # FTCS схема с периодическими граничными условиями
        for i in range(1, Nx - 1):
            u_new[i] = u[i] - sigma * (u[i+1] - u[i-1])
        
        # Периодические граничные условия
        u_new[0] = u[0] - sigma * (u[1] - u[Nx-2])
        u_new[Nx-1] = u_new[0]
        
        u = u_new
        u_history.append(u.copy())
        t_history.append((n + 1) * dt)
    
    return np.array(u_history), np.array(t_history)


# Решаем уравнение переноса методом FTCS
print("\n" + "="*60)
print("Решение методом FTCS")
print("="*60)

u_ftcs, t_ftcs = solve_advection_ftcs(u_initial, x, c, dt, dx, Nt)

# Визуализация нескольких моментов времени
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Схема FTCS: Forward Time, Centered Space', fontsize=14, fontweight='bold')

time_indices = [0, Nt//4, Nt//2, Nt]
for idx, ax in enumerate(axes.flat):
    t_idx = time_indices[idx]
    ax.plot(x, u_initial, 'k--', alpha=0.3, label='Начальное')
    ax.plot(x, u_ftcs[t_idx], 'r-', linewidth=2, label=f'FTCS')
    ax.set_xlabel('x, м')
    ax.set_ylabel('u')
    ax.set_title(f't = {t_ftcs[t_idx]:.2f} с')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.5, 2.0)

plt.tight_layout()
plt.show()


""" ## 3. Схема Upwind (против потока) ## """

def solve_advection_upwind(u0, x, c, dt, dx, Nt):
    """
    Решение уравнения переноса методом Upwind
    
    Схема (для c > 0): (u_i^{n+1} - u_i^n) / dt + c * (u_i^n - u_{i-1}^n) / dx = 0
    
    Эта схема устойчива при выполнении условия CFL: |c|*dt/dx ≤ 1
    Схема первого порядка точности, диссипативная (размывает профиль)
    
    Args:
        u0: начальное условие
        x: пространственная сетка
        c: скорость переноса
        dt: шаг по времени
        dx: шаг по пространству
        Nt: количество временных шагов
        
    Returns:
        u_history: история решения (Nt+1, Nx)
        t_history: временная сетка
    """
    Nx = len(x)
    u = u0.copy()
    u_history = [u.copy()]
    t_history = [0.0]
    
    sigma = c * dt / dx
    
    for n in range(Nt):
        u_new = u.copy()
        
        if c > 0:
            # Upwind для положительной скорости
            for i in range(1, Nx):
                u_new[i] = u[i] - sigma * (u[i] - u[i-1])
            # Периодическое граничное условие
            u_new[0] = u[0] - sigma * (u[0] - u[Nx-2])
        else:
            # Upwind для отрицательной скорости
            for i in range(Nx - 1):
                u_new[i] = u[i] - sigma * (u[i+1] - u[i])
            u_new[Nx-1] = u[Nx-1] - sigma * (u[0] - u[Nx-1])
        
        u = u_new
        u_history.append(u.copy())
        t_history.append((n + 1) * dt)
    
    return np.array(u_history), np.array(t_history)


# Решаем уравнение переноса методом Upwind
print("\n" + "="*60)
print("Решение методом Upwind")
print("="*60)

u_upwind, t_upwind = solve_advection_upwind(u_initial, x, c, dt, dx, Nt)

# Визуализация нескольких моментов времени
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Схема Upwind: стабильная, но диссипативная', fontsize=14, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    t_idx = time_indices[idx]
    ax.plot(x, u_initial, 'k--', alpha=0.3, label='Начальное')
    ax.plot(x, u_upwind[t_idx], 'b-', linewidth=2, label='Upwind')
    ax.set_xlabel('x, м')
    ax.set_ylabel('u')
    ax.set_title(f't = {t_upwind[t_idx]:.2f} с')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.5, 2.0)

plt.tight_layout()
plt.show()


""" ## 4. Схема Лакса-Вендроффа (Lax-Wendroff) ## """

def solve_advection_lax_wendroff(u0, x, c, dt, dx, Nt):
    """
    Решение уравнения переноса методом Лакса-Вендроффа
    
    Схема: u_i^{n+1} = u_i^n - (σ/2)(u_{i+1}^n - u_{i-1}^n) + (σ²/2)(u_{i+1}^n - 2u_i^n + u_{i-1}^n)
    где σ = c*dt/dx (число Куранта)
    
    Схема второго порядка точности, условно устойчива при |σ| ≤ 1
    Обладает низкой диссипацией, но проявляет дисперсию (осцилляции на разрывах)
    
    Args:
        u0: начальное условие
        x: пространственная сетка
        c: скорость переноса
        dt: шаг по времени
        dx: шаг по пространству
        Nt: количество временных шагов
        
    Returns:
        u_history: история решения (Nt+1, Nx)
        t_history: временная сетка
    """
    Nx = len(x)
    u = u0.copy()
    u_history = [u.copy()]
    t_history = [0.0]
    
    sigma = c * dt / dx
    
    for n in range(Nt):
        u_new = u.copy()
        
        # Лакс-Вендрофф схема
        for i in range(1, Nx - 1):
            u_new[i] = (u[i] 
                       - (sigma / 2) * (u[i+1] - u[i-1])
                       + (sigma**2 / 2) * (u[i+1] - 2*u[i] + u[i-1]))
        
        # Периодические граничные условия
        u_new[0] = (u[0] 
                   - (sigma / 2) * (u[1] - u[Nx-2])
                   + (sigma**2 / 2) * (u[1] - 2*u[0] + u[Nx-2]))
        u_new[Nx-1] = u_new[0]
        
        u = u_new
        u_history.append(u.copy())
        t_history.append((n + 1) * dt)
    
    return np.array(u_history), np.array(t_history)


# Решаем уравнение переноса методом Лакса-Вендроффа
print("\n" + "="*60)
print("Решение методом Лакса-Вендроффа")
print("="*60)

u_lax_wendroff, t_lax_wendroff = solve_advection_lax_wendroff(u_initial, x, c, dt, dx, Nt)

# Визуализация нескольких моментов времени
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Схема Лакса-Вендроффа: высокая точность с дисперсией', fontsize=14, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    t_idx = time_indices[idx]
    ax.plot(x, u_initial, 'k--', alpha=0.3, label='Начальное')
    ax.plot(x, u_lax_wendroff[t_idx], 'g-', linewidth=2, label='Lax-Wendroff')
    ax.set_xlabel('x, м')
    ax.set_ylabel('u')
    ax.set_title(f't = {t_lax_wendroff[t_idx]:.2f} с')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.5, 2.0)

plt.tight_layout()
plt.show()


""" ## 5. Схема Leap-frog (чехарда) ## """

def solve_advection_leapfrog(u0, x, c, dt, dx, Nt):
    """
    Решение уравнения переноса методом Leap-frog
    
    Схема: (u_i^{n+1} - u_i^{n-1}) / (2*dt) + c * (u_{i+1}^n - u_{i-1}^n) / (2*dx) = 0
    
    Трехслойная схема, требует два начальных слоя.
    Первый шаг выполняется методом Upwind или Lax-Wendroff.
    Схема второго порядка точности с минимальной диссипацией.
    
    Args:
        u0: начальное условие
        x: пространственная сетка
        c: скорость переноса
        dt: шаг по времени
        dx: шаг по пространству
        Nt: количество временных шагов
        
    Returns:
        u_history: история решения (Nt+1, Nx)
        t_history: временная сетка
    """
    Nx = len(x)
    u_history = [u0.copy()]
    t_history = [0.0]
    
    sigma = c * dt / dx
    
    # Первый шаг делаем методом Upwind для инициализации
    u_old = u0.copy()
    u_current = u0.copy()
    
    for i in range(1, Nx):
        u_current[i] = u_old[i] - sigma * (u_old[i] - u_old[i-1])
    u_current[0] = u_old[0] - sigma * (u_old[0] - u_old[Nx-2])
    
    u_history.append(u_current.copy())
    t_history.append(dt)
    
    # Остальные шаги методом Leap-frog
    for n in range(1, Nt):
        u_new = u_old.copy()
        
        # Leap-frog схема
        for i in range(1, Nx - 1):
            u_new[i] = u_old[i] - sigma * (u_current[i+1] - u_current[i-1])
        
        # Периодические граничные условия
        u_new[0] = u_old[0] - sigma * (u_current[1] - u_current[Nx-2])
        u_new[Nx-1] = u_new[0]
        
        # Сдвиг временных слоев
        u_old = u_current.copy()
        u_current = u_new.copy()
        
        u_history.append(u_current.copy())
        t_history.append((n + 1) * dt)
    
    return np.array(u_history), np.array(t_history)


# Решаем уравнение переноса методом Leap-frog
print("\n" + "="*60)
print("Решение методом Leap-frog")
print("="*60)

u_leapfrog, t_leapfrog = solve_advection_leapfrog(u_initial, x, c, dt, dx, Nt)

# Визуализация нескольких моментов времени
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Схема Leap-frog: минимальная диссипация', fontsize=14, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    t_idx = time_indices[idx]
    ax.plot(x, u_initial, 'k--', alpha=0.3, label='Начальное')
    ax.plot(x, u_leapfrog[t_idx], 'm-', linewidth=2, label='Leap-frog')
    ax.set_xlabel('x, м')
    ax.set_ylabel('u')
    ax.set_title(f't = {t_leapfrog[t_idx]:.2f} с')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.5, 2.0)

plt.tight_layout()
plt.show()


""" ## 6. Сравнение всех методов ## """

# Визуализация всех методов на одном графике
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Сравнение численных схем для уравнения переноса', fontsize=16, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    t_idx = time_indices[idx]
    ax.plot(x, u_initial, 'k--', linewidth=2, alpha=0.5, label='Начальное')
    ax.plot(x, u_ftcs[t_idx], 'r-', linewidth=1.5, alpha=0.8, label='FTCS')
    ax.plot(x, u_upwind[t_idx], 'b-', linewidth=1.5, alpha=0.8, label='Upwind')
    ax.plot(x, u_lax_wendroff[t_idx], 'g-', linewidth=1.5, alpha=0.8, label='Lax-Wendroff')
    ax.plot(x, u_leapfrog[t_idx], 'm-', linewidth=1.5, alpha=0.8, label='Leap-frog')
    ax.set_xlabel('x, м', fontsize=11)
    ax.set_ylabel('u', fontsize=11)
    ax.set_title(f't = {t_ftcs[t_idx]:.2f} с', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.5, 2.0)

plt.tight_layout()
plt.show()


""" ## 7. Анимация переноса волны ## """

def create_animation(x, u_histories, labels, colors, title):
    """
    Создание анимации для сравнения различных схем
    
    Args:
        x: пространственная сетка
        u_histories: список массивов решений
        labels: список названий схем
        colors: список цветов для каждой схемы
        title: заголовок анимации
        
    Returns:
        animation объект
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    lines = []
    for color, label in zip(colors, labels):
        line, = ax.plot([], [], color=color, linewidth=2, label=label)
        lines.append(line)
    
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.5, 2.0)
    ax.set_xlabel('x, м', fontsize=12)
    ax.set_ylabel('u', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        for line in lines:
            line.set_data([], [])
        time_text.set_text('')
        return lines + [time_text]
    
    def animate(frame):
        for line, u_history in zip(lines, u_histories):
            line.set_data(x, u_history[frame])
        time_text.set_text(f't = {frame * dt:.2f} с')
        return lines + [time_text]
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(u_histories[0]), 
                        interval=50, blit=True)
    
    plt.close()
    return anim


# Создаем анимацию со всеми схемами
print("\n" + "="*60)
print("Создание анимации...")
print("="*60)

anim = create_animation(
    x,
    [u_ftcs, u_upwind, u_lax_wendroff, u_leapfrog],
    ['FTCS', 'Upwind', 'Lax-Wendroff', 'Leap-frog'],
    ['red', 'blue', 'green', 'magenta'],
    'Сравнение численных схем: перенос волны в форме котика'
)

# Отображаем анимацию (в Jupyter)
HTML(anim.to_jshtml())


""" ## Выводы ## """

"""
### Основные результаты сравнения численных схем:

1. **FTCS (Forward Time, Centered Space)**
   - ❌ **Безусловно неустойчива** для гиперболических уравнений
   - Решение взрывается независимо от выбора шагов
   - Демонстрирует важность анализа устойчивости схем

2. **Upwind (против потока)**
   - ✅ **Устойчива** при выполнении условия CFL: $|c| \\Delta t / \\Delta x \\leq 1$
   - ⚠️ **Сильная численная диссипация** (вязкость)
   - Профиль размывается, острые края сглаживаются
   - Схема первого порядка точности
   - Подходит для задач, где важна стабильность, а точность второстепенна

3. **Lax-Wendroff (Лакс-Вендрофф)**
   - ✅ **Высокая точность** на гладких решениях
   - Схема второго порядка точности
   - ⚠️ **Численная дисперсия**: осцилляции вблизи разрывов
   - Почти полное отсутствие диссипации
   - Хороша для гладких волн, проблематична для разрывов

4. **Leap-frog (чехарда)**
   - ✅ **Минимальная диссипация**
   - Схема второго порядка точности
   - ⚠️ Присутствует дисперсия (как у Lax-Wendroff)
   - Требует три временных слоя (более затратна по памяти)
   - Широко используется в метеорологии

### Ключевые понятия:

- **Условие CFL (Куранта-Фридрихса-Леви)**: $\\sigma = |c| \\Delta t / \\Delta x \\leq 1$ 
  Физический смысл: за один временной шаг информация не должна проходить больше одной ячейки сетки

- **Численная диссипация**: искусственное размытие профиля, сглаживание острых краев

- **Численная дисперсия**: паразитные осцилляции, особенно вблизи разрывов или резких градиентов

### Рекомендации по выбору схемы:

- Для **простых задач**, где важна стабильность: **Upwind**
- Для **гладких решений** с высокими требованиями к точности: **Lax-Wendroff** или **Leap-frog**
- Для **задач с разрывами** (ударные волны): необходимы более продвинутые методы высокого разрешения (TVD, MUSCL, WENO)

### Взгляд в будущее:

Для реальных задач газовой динамики, астрофизики и других областей с ударными волнами 
используются консервативные схемы высокого разрешения, которые комбинируют:
- Точность схем второго порядка на гладких участках
- Монотонность и отсутствие осцилляций на разрывах
- Консервативность (сохранение интегральных инвариантов)
"""

