import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BurgersEquation:
    def __init__(self, L=1.0, Nx=200, T=1.0, Nt=500):
        """
        Инициализация параметров
        
        Parameters:
        L - длина области
        Nx - количество точек по пространству
        T - конечное время
        Nt - количество шагов по времени
        """
        self.L = L
        self.Nx = Nx
        self.T = T
        self.Nt = Nt
        
        # Пространственная сетка
        self.x = np.linspace(0, L, Nx)
        self.dx = self.x[1] - self.x[0]
        
        # Временная сетка
        self.dt = T / Nt
        
        # Начальное условие
        self.u0 = np.sin(2 * np.pi * self.x)
        
        # Коэффициент вязкости (равен 0 для невязкого случая)
        self.nu = 0.0
        
    def upwind_scheme(self, u):
        """
        Upwind схема для уравнения Бюргерса
        """
        u_new = np.zeros_like(u)
        
        for i in range(1, self.Nx-1):
            # Определение направления потока
            if u[i] >= 0:
                # Поток слева направо
                u_new[i] = u[i] - (self.dt/self.dx) * u[i] * (u[i] - u[i-1])
            else:
                # Поток справа налево
                u_new[i] = u[i] - (self.dt/self.dx) * u[i] * (u[i+1] - u[i])
        
        # Периодические граничные условия
        u_new[0] = u_new[-2]
        u_new[-1] = u_new[1]
        
        return u_new
    
    def lax_wendroff_scheme(self, u):
        """
        Схема Лакса-Вендроффа для уравнения Бюргерса
        """
        u_new = np.zeros_like(u)
        
        for i in range(1, self.Nx-1):
            # Потоки на полуцелых шагах
            F_i_plus_half = 0.5 * (u[i+1]**2 + u[i]**2) / 2 - \
                           (self.dt/(2*self.dx)) * (u[i+1]**2/2 - u[i]**2/2)**2
            
            F_i_minus_half = 0.5 * (u[i]**2 + u[i-1]**2) / 2 - \
                            (self.dt/(2*self.dx)) * (u[i]**2/2 - u[i-1]**2/2)**2
            
            u_new[i] = u[i] - (self.dt/self.dx) * (F_i_plus_half - F_i_minus_half)
        
        # Периодические граничные условия
        u_new[0] = u_new[-2]
        u_new[-1] = u_new[1]
        
        return u_new
    
    def solve(self):
        """
        Решение уравнения Бюргерса обоими методами
        """
        # Инициализация массивов для решений
        self.u_upwind = np.zeros((self.Nt+1, self.Nx))
        self.u_lax_wendroff = np.zeros((self.Nt+1, self.Nx))
        
        # Начальные условия
        self.u_upwind[0] = self.u0.copy()
        self.u_lax_wendroff[0] = self.u0.copy()
        
        # Интегрирование по времени
        for n in range(self.Nt):
            self.u_upwind[n+1] = self.upwind_scheme(self.u_upwind[n])
            self.u_lax_wendroff[n+1] = self.lax_wendroff_scheme(self.u_lax_wendroff[n])
        
        return self.u_upwind, self.u_lax_wendroff

def plot_solutions(burgers, save_animation=False):
    """
    Визуализация решений
    """
    u_upwind, u_lax_wendroff = burgers.solve()
    
    # Выбор моментов времени для отображения
    time_indices = [0, burgers.Nt//4, burgers.Nt//2, 3*burgers.Nt//4, burgers.Nt]

    # Анимация
    if save_animation:
        create_animation(burgers, u_upwind, u_lax_wendroff)
    
    plt.figure(figsize=(15, 10))
    
    for i, t_idx in enumerate(time_indices):
        time = t_idx * burgers.dt
        
        plt.subplot(2, 3, i+1)
        plt.plot(burgers.x, u_upwind[t_idx], 'b-', linewidth=2, label='Upwind')
        plt.plot(burgers.x, u_lax_wendroff[t_idx], 'r--', linewidth=2, label='Lax-Wendroff')
        plt.plot(burgers.x, burgers.u0, 'k:', linewidth=1, label='Начальное условие')
        
        plt.title(f'Время t = {time:.2f}')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Установка одинаковых пределов для всех графиков
        plt.ylim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.show()

def create_animation(burgers, u_upwind, u_lax_wendroff):
    """
    Создание анимации эволюции решения
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    def animate(frame):
        # Каждый 10-й кадр для ускорения анимации
        n = frame * 10
        if n >= burgers.Nt:
            n = burgers.Nt - 1
        
        time = n * burgers.dt
        
        ax1.clear()
        ax2.clear()
        
        # График Upwind схемы
        ax1.plot(burgers.x, u_upwind[n], 'b-', linewidth=2)
        ax1.plot(burgers.x, burgers.u0, 'k:', linewidth=1, alpha=0.5)
        ax1.set_title(f'Upwind схема, t = {time:.3f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x,t)')
        ax1.set_ylim(-1.2, 1.2)
        ax1.grid(True, alpha=0.3)
        
        # График Лакса-Вендроффа
        ax2.plot(burgers.x, u_lax_wendroff[n], 'r-', linewidth=2)
        ax2.plot(burgers.x, burgers.u0, 'k:', linewidth=1, alpha=0.5)
        ax2.set_title(f'Lax-Wendroff схема, t = {time:.3f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('u(x,t)')
        ax2.set_ylim(-1.2, 1.2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    frames = burgers.Nt // 10
    anim = FuncAnimation(fig, animate, frames=frames, interval=50)
    
    # Сохранение анимации
    #anim.save('burgers_equation.gif', writer='pillow', fps=20)
    plt.show()

def compare_near_discontinuity(burgers):
    """
    Сравнение методов вблизи разрыва
    """
    u_upwind, u_lax_wendroff = burgers.solve()
    
    # Находим момент времени, когда образуется разрыв (примерно)
    t_shock = burgers.Nt // 2
    
    plt.figure(figsize=(12, 8))
    
    # Общий вид
    plt.subplot(2, 1, 1)
    plt.plot(burgers.x, u_upwind[t_shock], 'b-', linewidth=2, label='Upwind')
    plt.plot(burgers.x, u_lax_wendroff[t_shock], 'r--', linewidth=2, label='Lax-Wendroff')
    plt.title(f'Сравнение методов при t = {t_shock * burgers.dt:.3f}')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Увеличенный вид вблизи разрыва
    plt.subplot(2, 1, 2)
    
    # Находим область с максимальным градиентом
    grad_upwind = np.abs(np.gradient(u_upwind[t_shock]))
    shock_region = np.where(grad_upwind > 0.5 * np.max(grad_upwind))[0]
    
    if len(shock_region) > 0:
        center = shock_region[len(shock_region)//2]
        left = max(center - 20, 0)
        right = min(center + 20, burgers.Nx-1)
        
        plt.plot(burgers.x[left:right], u_upwind[t_shock, left:right], 'b-o', 
                markersize=4, linewidth=2, label='Upwind')
        plt.plot(burgers.x[left:right], u_lax_wendroff[t_shock, left:right], 'r--s', 
                markersize=4, linewidth=2, label='Lax-Wendroff')
        plt.title('Увеличенный вид вблизи разрыва')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Параметры расчета
    L = 1.0           # Длина области
    Nx = 200          # Количество точек по пространству
    T = 0.5           # Конечное время (увеличено для наблюдения разрыва)
    Nt = 500          # Количество шагов по времени
    
    # Создание объекта уравнения Бюргерса
    burgers = BurgersEquation(L=L, Nx=Nx, T=T, Nt=Nt)
    
    # Построение решений
    plot_solutions(burgers, save_animation=True)
    
    # Детальное сравнение вблизи разрыва
    compare_near_discontinuity(burgers)
    
    # Анализ численной диссипации и осцилляций
    u_upwind, u_lax_wendroff = burgers.solve()
    
    # Вычисление полной вариации для анализа осцилляций
    TV_upwind = np.sum(np.abs(np.diff(u_upwind, axis=1)), axis=1)
    TV_lax_wendroff = np.sum(np.abs(np.diff(u_lax_wendroff, axis=1)), axis=1)
    
    plt.figure(figsize=(10, 6))
    time = np.linspace(0, T, Nt+1)
    plt.plot(time, TV_upwind, 'b-', label='Upwind (полная вариация)')
    plt.plot(time, TV_lax_wendroff, 'r--', label='Lax-Wendroff (полная вариация)')
    plt.xlabel('Время')
    plt.ylabel('Полная вариация')
    plt.title('Эволюция полной вариации решений')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
