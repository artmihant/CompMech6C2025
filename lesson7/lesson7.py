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
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

""" ## Параметры задачи ## """

# Пространственная область
L = 10.0          # длина области, м
Nx = 1000          # количество узлов сетки
dx = L / (Nx - 1) # шаг по пространству

# Временные параметры
c = 1.0           # скорость переноса, м/с
T = 10.0           # время симуляции, с
dt = 0.00001         # шаг по времени, с 
Nt = int(T / dt)  # количество временных шагов

# Число Куранта (CFL number)
sigma = c * dt / dx
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
    
    # plt.close()
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

# plt.figure(figsize=(10, 4))
# plt.plot(x, u0, 'b-', linewidth=2, label='Синус')
# plt.xlabel('x, м')
# plt.ylabel('u')
# plt.title('Начальное условие')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()

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

# anim = create_animation(x, [
#     (u_ftcs, 'ftcs', 'red'),
#     (u_upwind, 'upwind', 'blue'), 
#     (u_lax_wendroff, 'lax_wendroff', 'orange')
# ])
# HTML(anim.to_jshtml())

""" ## 4. Неявная upwind-схема (Backward Euler + Upwind) ## """

def solve_advection_implicit_upwind(u0, c, Nx, dx, Nt, dt):
    """
    Решение уравнения переноса неявной upwind-схемой

    Схема (для c > 0): (u_i^{n+1} - u_i^n)/dt + c * (u_i^{n+1} - u_{i-1}^{n+1}) / dx = 0

    - Неусловно устойчива
    - Монотонна, но диссипативна (размывает фронты)
    - Периодические граничные условия реализованы через разреженную матрицу с угловым элементом

    Args:
        u0: начальное условие
        c: скорость переноса (предполагаем c > 0)
        Nx: количество узлов сетки
        dx: шаг по пространству
        Nt: количество временных шагов
        dt: шаг по времени

    Returns:
        u_history: история решения (Nt+1, Nx)
    """

    u_history = np.zeros((Nt+1, Nx))
    u_history[0] = u0

    sigma = c * dt / dx

    # Матрица A: (1+sigma) на диагонали, -sigma на поддиагонали и угловой элемент A[0, N-1] = -sigma
    main_diag = np.full(Nx, 1.0 + sigma)
    lower_diag = np.full(Nx - 1, -sigma)

    A = diags([main_diag, lower_diag], [0, -1], shape=(Nx, Nx), format='lil')
    A[0, Nx - 1] = -sigma
    A = A.tocsc()

    solve = factorized(A)

    for tau in range(Nt):
        u_history[tau + 1] = solve(u_history[tau])

    return u_history

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


""" ## 6. Схема Кранка–Николсона (усреднение FTCS и BTCS) ## """

def solve_advection_crank_nicolson(u0, c, Nx, dx, Nt, dt):
    """
    Решение уравнения переноса схемой Кранка–Николсона

    (u^{n+1} - u^n)/dt + c * ( (\partial_x u^{n+1} + \partial_x u^n) / 2 ) = 0,
    где производная по x аппроксимируется центральной разностью.

    Приводит к СЛАУ A u^{n+1} = b,
    A = I + (σ/4) S_{+1} - (σ/4) S_{-1},
    b = u^n - (σ/4) (S_{+1} u^n - S_{-1} u^n),
    где S_{±1} — циклические сдвиги на ±1 с периодическими ГУ, σ = c·dt/dx.

    Args:
        u0: начальное условие
        c: скорость переноса
        Nx: число узлов по x
        dx: шаг по x
        Nt: число шагов по времени
        dt: шаг по времени

    Returns:
        u_history: массив формы (Nt+1, Nx) с историей решения
    """

    u_history = np.zeros((Nt + 1, Nx))
    u_history[0] = u0

    sigma = c * dt / dx

    # Матрица A: diag=1, верхняя диагональ = +σ/4, нижняя диагональ = -σ/4
    main_diag = np.ones(Nx)
    upper_diag = np.full(Nx - 1, sigma / 4.0)
    lower_diag = np.full(Nx - 1, -sigma / 4.0)

    A = diags([main_diag, upper_diag, lower_diag], [0, 1, -1], shape=(Nx, Nx), format='lil')
    # Периодические ГУ (угловые элементы)
    A[0, Nx - 1] = -sigma / 4.0   # соответствует сдвигу -1
    A[Nx - 1, 0] = sigma / 4.0    # соответствует сдвигу +1
    A = A.tocsc()

    solve = factorized(A)

    for tau in range(Nt):
        u_prev = u_history[tau]
        rhs = u_prev - (sigma / 4.0) * (np.roll(u_prev, -1) - np.roll(u_prev, 1))
        u_history[tau + 1] = solve(rhs)

    return u_history

""" ## 7. Полунеявный Кранка–Николсона с upwind-потоком ## """

def solve_advection_cn_upwind(u0, c, Nx, dx, Nt, dt):
    """
    Полунеявная схема: усреднение явной upwind и неявной upwind

    (u^{n+1} - u^n)/dt + c * [ (D_up u^{n+1}) + (D_up u^n) ] / 2 = 0

    Для c>0 оператор upwind: (u_i - u_{i-1})/dx.
    Итоговая СЛАУ: A u^{n+1} = b,
      A = I + (σ/2) * (I - S_{-1}),
      b = u^n - (σ/2) * (I - S_{-1}) u^n,
    где σ = c·dt/dx, S_{-1} — циклический сдвиг на -1.
    """

    u_history = np.zeros((Nt + 1, Nx))
    u_history[0] = u0

    sigma = c * dt / dx

    # Матрица A для c>0: diag = 1 + σ/2, поддиагональ = -σ/2, угловой A[0,N-1] = -σ/2
    main_diag = np.full(Nx, 1.0 + sigma / 2.0)
    lower_diag = np.full(Nx - 1, -sigma / 2.0)

    A = diags([main_diag, lower_diag], [0, -1], shape=(Nx, Nx), format='lil')
    A[0, Nx - 1] = -sigma / 2.0
    A = A.tocsc()

    solve = factorized(A)

    for tau in range(Nt):
        u_prev = u_history[tau]
        rhs = u_prev - (sigma / 2.0) * (u_prev - np.roll(u_prev, 1))
        u_history[tau + 1] = solve(rhs)

    return u_history

""" ## 8. Метод линий (MOL) + RK4 ## """

def solve_advection_mol_rk4(u0, c, Nx, dx, Nt, dt):
    """
    Метод линий: дискретизация по x (центральная разность), интегрирование по t методом RK4

    du/dt = -c * d(u)/dx,   d(u)/dx ≈ (u_{i+1} - u_{i-1}) / (2*dx)
    Периодические граничные условия реализованы через циклический сдвиг (np.roll).

    Args:
        u0: начальное условие (Nx,)
        c: скорость переноса
        Nx: число узлов по x
        dx: шаг по x
        Nt: число шагов по времени
        dt: шаг по времени

    Returns:
        u_history: массив формы (Nt+1, Nx)
    """

    def compute_du_dt(u):
        return -c * (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)

    u_history = np.zeros((Nt + 1, Nx))
    u_history[0] = u0

    for tau in range(Nt):
        u = u_history[tau]
        k1 = compute_du_dt(u)
        k2 = compute_du_dt(u + 0.5 * dt * k1)
        k3 = compute_du_dt(u + 0.5 * dt * k2)
        k4 = compute_du_dt(u + dt * k3)
        u_history[tau + 1] = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return u_history

anim = create_animation(x, [
    (solve_advection_ftcs(u0, c, Nx, dx, Nt, dt), 'ftcs', 'red'),
    # (solve_advection_upwind(u0, c, Nx, dx, Nt, dt), 'upwind', 'blue'), 
    (solve_advection_lax_wendroff(u0, c, Nx, dx, Nt, dt), 'lax_wendroff', 'orange'),
    # (solve_advection_implicit_upwind(u0, c, Nx, dx, Nt, dt), 'implicit_upwind', 'purple'),
    # (solve_advection_crank_nicolson(u0, c, Nx, dx, Nt, dt), 'crank_nicolson', 'black'),
    # (solve_advection_cn_upwind(u0, c, Nx, dx, Nt, dt), 'cn_upwind', 'brown'),
    # (solve_advection_mol_rk4(u0, c, Nx, dx, Nt, dt), 'mol_rk4', 'magenta'),
    # (solve_advection_leapfrog(u0, c, Nx, dx, Nt, dt), 'leapfrog', 'green')
])

plt.show()
# HTML(anim.to_jshtml())