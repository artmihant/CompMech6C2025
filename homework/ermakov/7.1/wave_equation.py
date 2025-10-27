import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


def triangular_pulse(x, L, center=0.5, width=0.2):
    """
    Треугольный профиль для начального условия 'щипок струны'
    """
    height = 1.0
    left = center - width/2
    right = center + width/2
    
    condition = (x >= left) & (x <= right)
    y = np.zeros_like(x)
    
    # Левая половина треугольника
    left_half = (x >= left) & (x <= center)
    y[left_half] = height * (x[left_half] - left) / (center - left)
    
    # Правая половина треугольника
    right_half = (x > center) & (x <= right)
    y[right_half] = height * (right - x[right_half]) / (right - center)
    
    return y


def solve_wave_equation(L=1.0, c=1.0, T=5.0, nx=101, CFL=0.8, 
                       initial_shape=None, initial_velocity=None):
    """
    Решение волнового уравнения явной схемой 'крест'
    """
    # Параметры сетки
    dx = L / (nx - 1)
    dt = CFL * dx / c
    nt = int(T / dt) + 1
    
    # Временные шаги для сохранения анимации
    save_every = max(1, nt // 200)
    
    # Пространственная сетка
    x = np.linspace(0, L, nx)
    
    # Инициализация решения
    u_prev = np.zeros(nx)  # u^{n-1}
    u_curr = np.zeros(nx)  # u^n
    u_next = np.zeros(nx)  # u^{n+1}
    
    # Начальные условия
    if initial_shape is not None:
        u_curr = initial_shape(x)
    
    # Первый шаг по времени (используем начальную скорость)
    u_prev = u_curr.copy()
    if initial_velocity is not None:
        u_prev = u_curr - dt * initial_velocity(x)
    else:
        # Аппроксимация для g(x) = 0
        for i in range(1, nx-1):
            u_prev[i] = u_curr[i] + 0.5 * (CFL**2) * (u_curr[i+1] - 2*u_curr[i] + u_curr[i-1])
    
    # Списки для хранения результатов
    time_steps = []
    solutions = []
    times = []
    
    # Основной цикл по времени
    for n in range(nt):
        current_time = n * dt
        
        # Сохраняем каждый save_every-ый шаг
        if n % save_every == 0:
            solutions.append(u_curr.copy())
            times.append(current_time)
            time_steps.append(n)
        
        # Вычисление следующего временного слоя
        for i in range(1, nx-1):
            u_next[i] = 2*u_curr[i] - u_prev[i] + \
                       (CFL**2) * (u_curr[i+1] - 2*u_curr[i] + u_curr[i-1])
        
        # Граничные условия
        u_next[0] = 0
        u_next[-1] = 0
        
        # Обновление для следующей итерации
        u_prev, u_curr = u_curr, u_next.copy()
    
    return x, np.array(solutions), np.array(times), dt, dx, CFL


def create_animation(x, solutions, times, CFL):
    """
    Создание анимации колебаний струны
    """
    fig, ax = plt.subplots(figsize=(18, 12))
    plt.subplots_adjust(bottom=0.2)
    
    line, = ax.plot(x, solutions[0], 'b-', linewidth=2, label=f'CFL = {CFL}')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Положение x')
    ax.set_ylabel('Амплитуда u(x,t)')
    ax.set_title('Колебания струны: явная схема "крест"')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def animate_frame(i):
        line.set_ydata(solutions[i])
        time_text.set_text(f'Время: {times[i]:.3f} с\nШаг: {i}')
        return line, time_text
    
    anim = FuncAnimation(fig, animate_frame, frames=len(solutions),
                        interval=50, blit=True)
    
    return fig, anim

def compare_CFL():
    """
    Сравнение решений для разных значений CFL
    """
    CFL_values = [0.5, 0.8, 1.0, 1.2]
    L, c, T = 1.0, 1.0, 2.0
    nx = 101
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, CFL in enumerate(CFL_values):
        try:
            x, solutions, times, dt, dx, _ = solve_wave_equation(
                L=L, c=c, T=T, nx=nx, CFL=CFL,
                initial_shape=lambda x: triangular_pulse(x, L)
            )
            
            # Показываем несколько моментов времени
            time_indices = [0, len(solutions)//4, len(solutions)//2, 3*len(solutions)//4, -1]
            colors = ['red', 'orange', 'green', 'blue', 'purple']
            labels = [f't={times[i]:.2f}' for i in time_indices]
            
            j = 0
            for i, color in zip(time_indices, colors):
                axes[idx].plot(x, solutions[i], color=color, linewidth=2, 
                              label=labels[j])
                j = j + 1
            
            axes[idx].set_title(f'CFL = {CFL} (dt={dt:.4f})')
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('u(x,t)')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()
            axes[idx].set_ylim(-1.2, 1.2)
        
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Ошибка:\n{str(e)}', 
                          transform=axes[idx].transAxes, ha='center')
            axes[idx].set_title(f'CFL = {CFL} - ОШИБКА')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":    
    L, c = 1.0, 1.0  # длина струны, скорость волны 
    T = 4.0
    nx = 201 # число узлов
    CFL = 0.8 # число Куранта
    
    print("\nВыполнение расчета...")
    x, solutions, times, dt, dx, CFL_actual = solve_wave_equation(
        L=L, c=c, T=T, nx=nx, CFL=CFL,
        initial_shape=lambda x: triangular_pulse(x, L)
    )
    
    print("\nСоздание анимации...")
    fig, anim = create_animation(x, solutions, times, CFL)
    plt.show()

    print("\nСравнение разных значений CFL...")
    compare_CFL()
