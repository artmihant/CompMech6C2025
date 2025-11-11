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
T = 30.0           # время симуляции, с
dt = 0.001         # шаг по времени, с 
Nt = int(T / dt)  # количество временных шагов

# Число Куранта (CFL number)
sigma = c * dt / dx
sigma2 = c * dt / (dx**2)/2
# Сетка
x = np.linspace(0, L, Nx)

# Число кадров анимации и число dt на кадр
fps = 20 # кадров в секунду
slowmo = 1 # отношение времени анимации к времени симуляции

frames = int(T * fps * slowmo) 
framerate = int(Nt/frames) or 1

print(f"Параметры численной схемы:")
print(f"  Шаг по пространству dx = {dx:.4f} м")
print(f"  Шаг по времени dt = {dt:.4f} с")
print(f"  Число Куранта σ = c·dt/dx = {sigma:.4f}")
print(f"  Число Куранта σ2 = c·dt/(dx**2)/2 = {sigma2:.4f}")
print(f"  Условие устойчивости CFL: σ ≤ 1.0")
print(f"  Количество временных шагов: {Nt}")

def create_animation(x, graphs):
    """
    Создание анимации для сравнения различных схем
    
    Args:
        x: пространственная сетка
        u_histories: список массивов решений
        labels: список названий схем
        colors: список цветов для каждой схемы
        
    Returns:
        animation объект
    """
    fig, ax = plt.subplots(figsize=(6, 3))


    lines = []
    for u_history, label, color in graphs:
        line, = ax.plot([], [], color=color, linewidth=2, label=label)
        lines.append(line)
    
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.5, 2.0)
    ax.set_xlabel('x, м', fontsize=12)
    ax.set_ylabel('u', fontsize=12)
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
    
    def animate(i):
        for (u_history, label, color), line in zip(graphs, lines):
            line.set_data(x, u_history[framerate*i])
        time_text.set_text(f't = {framerate*i*dt:.2f} с')
        return lines + [time_text]
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=frames, 
                        interval=1000/fps, blit=True)
    
    plt.close()
    return anim

""" ## Начальное условие ## """

def initial_condition(x, x_center=3.0, width=1.0, height=1.0):

    # u = 0.3*np.sin(2*np.pi*x/L)
    # return u

    u = np.zeros_like(x)

    body_mask = np.abs(x - x_center) <= width / 2
    x_body = x[body_mask]
    radius = width / 2
    u[body_mask] = height * np.sqrt(1 - ((x_body - x_center) / radius)**2)
    
    left_ear_x = x_center - width * 0.35
    left_ear_width = width * 0.15
    left_ear_mask = np.abs(x - left_ear_x) <= left_ear_width
    x_left_ear = x[left_ear_mask]
    u[left_ear_mask] = np.maximum(u[left_ear_mask], 
                                    height * 1.3 * (1 - np.abs(x_left_ear - left_ear_x) / left_ear_width))
    
    right_ear_x = x_center + width * 0.35
    right_ear_width = width * 0.15
    right_ear_mask = np.abs(x - right_ear_x) <= right_ear_width
    x_right_ear = x[right_ear_mask]
    u[right_ear_mask] = np.maximum(u[right_ear_mask], 
                                     height * 1.3 * (1 - np.abs(x_right_ear - right_ear_x) / right_ear_width))
    
    return u

u0 = initial_condition(x)

plt.figure(figsize=(10, 4))
plt.plot(x, u0, 'b-', linewidth=2, label='Синус')
plt.xlabel('x, м')
plt.ylabel('u')
plt.title('Начальное условие')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

""" ## 1. Схема FTCS (Forward Time, Centered Space) ## """

def solve_advection_ftcs(u0, c, Nx, dx, Nt, dt):
    """
    Решение уравнения переноса методом FTCS

    Схема неустойчива всегда

    Args:
        u0: начальное условие
        c: скорость переноса
        Nx: количество узлов сетки
        dx: шаг по пространству
        Nt: количество временных шагов
        dt: шаг по времени
        
    Returns:
        u_history: история решения (Nt+1, Nx)
    """

    u_history = np.zeros((Nt+1, Nx))

    u_history[0] = u0
    
    sigma =  c * dt / dx
    
    for tau in range(Nt):
        
        u_history[tau+1] = u_history[tau] - sigma/2 * (np.roll(u_history[tau],-1) - np.roll(u_history[tau],1))
        
        # for xi in range(-1,Nx-1):
        #     u_history[tau+1, xi] = u_history[tau, xi] - sigma/2 * (u_history[tau,xi+1] - u_history[tau,xi-1])

    return np.array(u_history)

# Решаем уравнение переноса методом FTCS

u_ftcs = solve_advection_ftcs(u0, c, Nx, dx, Nt, dt)

# anim = create_animation(x, [
#     (u_ftcs, 'ftcs', 'red')
# ])
# HTML(anim.to_jshtml())

""" ## 2. Схема Upwind (против ветра) ## """

def solve_advection_upwind(u0, c, Nx, dx, Nt, dt):
    """
    Решение уравнения переноса методом Upwind
    
    Схема (для c > 0): (u_i^{n+1} - u_i^n) / dt + c * (u_i^n - u_{i-1}^n) / dx = 0
    
    Эта схема устойчива при выполнении условия CFL: |c|*dt/dx ≤ 1
    Схема первого порядка точности, диссипативная (размывает профиль)
    
    Args:
        u0: начальное условие
        c: скорость переноса
        Nx: количество узлов сетки
        dx: шаг по пространству
        Nt: количество временных шагов
        dt: шаг по времени
        
    Returns:
        u_history: история решения (Nt+1, Nx)
    """

    u_history = np.zeros((Nt+1, Nx))

    u_history[0] = u0
    
    sigma =  c * dt / dx
    
    for tau in range(Nt):
        
        u_history[tau+1] = u_history[tau] - sigma * (u_history[tau] - np.roll(u_history[tau],1))
        
        # for xi in range(-1,Nx-1):
        #     u_history[tau+1, xi] = u_history[tau, xi] - sigma/2 * (u_history[tau,xi+1] - u_history[tau,xi-1])

    return np.array(u_history)

u_upwind = solve_advection_upwind(u0, c, Nx, dx, Nt, dt)

# anim = create_animation(x, [
#     (u_ftcs, 'ftcs', 'red'),
#     (u_upwind, 'upwind', 'blue')
# ])
# HTML(anim.to_jshtml())

""" ## 3. Схема Лакса-Вендроффа (Lax-Wendroff) ## """

def solve_advection_lax_wendroff(u0, c, Nx, dx, Nt, dt):
    """
    Решение уравнения переноса методом Лакса-Вендроффа
    
    Схема: u_i^{n+1} = u_i^n - (σ/2)(u_{i+1}^n - u_{i-1}^n) + (σ²/2)(u_{i+1}^n - 2u_i^n + u_{i-1}^n)
    где σ = c*dt/dx (число Куранта)
    
    Схема второго порядка точности, условно устойчива при |σ| ≤ 1
    Обладает низкой диссипацией, но проявляет дисперсию (осцилляции на разрывах)
    
    Args:
        u0: начальное условие
        c: скорость переноса
        Nx: количество узлов сетки
        dx: шаг по пространству
        Nt: количество временных шагов
        dt: шаг по времени
        
    Returns:
        u_history: история решения (Nt+1, Nx)
    """

    u_history = np.zeros((Nt+1, Nx))

    u_history[0] = u0
    
    sigma =  c * dt / dx
    
    for tau in range(Nt):
        
        # u_history[tau+1,1:-1] = u_history[tau,1:-1] - 2 *sigma * (u_history[tau,2:] - u_history[tau,:-2])
        
        u_history[tau+1] = u_history[tau] \
            - (sigma/2) * (np.roll(u_history[tau],-1) - np.roll(u_history[tau],1)) \
            + (sigma**2 / 2) * (np.roll(u_history[tau], -1) - 2*u_history[tau] + np.roll(u_history[tau], 1))

        # for xi in range(-1,Nx-1):
        #     u_history[tau+1, xi] = u_history[tau, xi] \
        #         - sigma/2 * (u_history[tau,xi+1] - u_history[tau,xi-1]) \
        #         + (sigma**2 / 2) * (u_history[tau,xi+1] - 2*u_history[tau, xi] + u_history[tau,xi-1])
       
    
    return np.array(u_history)

u_lax_wendroff = solve_advection_lax_wendroff(u0, c, Nx, dx, Nt, dt)

# anim = create_animation(x, [
#     (u_ftcs, 'ftcs', 'red'),
#     (u_upwind, 'upwind', 'blue'), 
#     (u_lax_wendroff, 'lax_wendroff', 'orange')
# ])
# HTML(anim.to_jshtml())

""" ## 5. Схема Leap-frog (чехарда) ## """

def solve_advection_leapfrog(u0, c, Nx, dx, Nt, dt):
    """
    Решение уравнения переноса методом Leap-frog
    
    Схема: (u_i^{n+1} - u_i^{n-1}) / (2*dt) + c * (u_{i+1}^n - u_{i-1}^n) / (2*dx) = 0
    
    Трехслойная схема, требует два начальных слоя.
    Первый шаг выполняется методом Upwind или Lax-Wendroff.
    Схема второго порядка точности с минимальной диссипацией.
    
    Args:
        u0: начальное условие
        c: скорость переноса
        Nx: количество узлов сетки
        dx: шаг по пространству
        Nt: количество временных шагов
        dt: шаг по времени
        
    Returns:
        u_history: история решения (Nt+1, Nx)
    """


    u_history = np.zeros((Nt+1, Nx))

    u_history[0] = u0
    
    sigma =  c * dt / dx
    
    u_history[1] = u_history[0] - sigma * (u_history[0] - np.roll(u_history[0],1))

    # Остальные шаги методом Leap-frog
    for tau in range(1, Nt):    

        u_history[tau+1] = u_history[tau-1] - sigma * (np.roll(u_history[tau], -1) - np.roll(u_history[tau], +1))
        
    return u_history

u_leapfrog = solve_advection_leapfrog(u0, c, Nx, dx, Nt, dt)

anim = create_animation(x, [
    (u_ftcs, 'ftcs', 'red'),
    (u_upwind, 'upwind', 'blue'), 
    (u_lax_wendroff, 'lax_wendroff', 'orange'),
    (u_leapfrog, 'leapfrog', 'green')

])
HTML(anim.to_jshtml())

