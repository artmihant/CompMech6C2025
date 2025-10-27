import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ShallowWaterSolver:
    def __init__(self, L=100.0, H=10.0, g=9.81, nx=200, cfl=0.8):
        """
        Инициализация решателя уравнений мелкой воды
        
        Parameters:
        L - длина области
        H - средняя глубина
        g - ускорение свободного падения
        nx - количество точек сетки
        cfl - число Куранта для устойчивости
        """
        self.L = L
        self.H = H
        self.g = g
        self.nx = nx
        self.cfl = cfl
        
        # Скорость гравитационных волн
        self.c = np.sqrt(g * H)
        
        # Сетка
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)
        
        # Временной шаг из условия CFL
        self.dt = cfl * self.dx / self.c
        
        # Поля: высота и скорость
        self.eta = np.zeros(nx)  # возмущение высоты η = h - H
        self.u = np.zeros(nx)    # скорость
        
        # Консервативные переменные
        self.U = np.zeros((2, nx))  # [h, hu]
        
    def initial_condition(self, amplitude=0.5, width=5.0):
        """Начальное условие - гауссов 'горб' в центре"""
        center = self.L / 2
        self.eta = amplitude * np.exp(-((self.x - center) / width)**2)
        self.u = np.zeros_like(self.eta)
        
        # Консервативные переменные
        self.U[0, :] = self.H + self.eta  # h = H + η
        self.U[1, :] = self.U[0, :] * self.u  # hu
        
    def lax_friedrichs_step(self):
        """Один шаг по схеме Лакса-Фридрихса"""
        U_new = np.zeros_like(self.U)
        
        # Потоки F = [hu, hu² + ½gh²]
        F = np.zeros_like(self.U)
        F[0, :] = self.U[1, :]  # hu
        F[1, :] = (self.U[1, :]**2 / self.U[0, :]) + 0.5 * self.g * self.U[0, :]**2  # hu² + ½gh²
        
        # Схема Лакса-Фридрихса
        for i in range(1, self.nx - 1):
            U_new[:, i] = 0.5 * (self.U[:, i+1] + self.U[:, i-1]) - \
                        0.5 * self.dt / self.dx * (F[:, i+1] - F[:, i-1])
        
        # Граничные условия (свободного истечения)
        U_new[:, 0] = U_new[:, 1]
        U_new[:, -1] = U_new[:, -2]
        
        self.U = U_new.copy()
        
        # Восстановление физических переменных
        self.eta = self.U[0, :] - self.H
        self.u = self.U[1, :] / self.U[0, :]
        
    def solve(self, T_max=20.0):
        """Решение уравнений на временном интервале [0, T_max]"""
        n_steps = int(T_max / self.dt)
        times = []
        solutions = []
        
        for step in range(n_steps):
            if step % 10 == 0:  # Сохраняем каждые 10 шагов
                times.append(step * self.dt)
                solutions.append(self.eta.copy())
            
            self.lax_friedrichs_step()
            
        return times, solutions

def animate_solution():
    """Анимация распространения волны"""
    solver = ShallowWaterSolver(L=100.0, H=10.0, nx=400, cfl=0.5)
    solver.initial_condition(amplitude=1.0, width=3.0)
    
    # Создание анимации
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot(solver.x, solver.eta, 'b-', linewidth=2, label='η(x,t)')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlim(0, solver.L)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Положение, x')
    ax.set_ylabel('Возмущение высоты, η')
    ax.set_title('Распространение гравитационных волн в мелкой воде')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    def animate(frame):
        for _ in range(5):  # Несколько шагов между кадрами
            solver.lax_friedrichs_step()
        
        line.set_ydata(solver.eta)
        time_text.set_text(f'Время: {solver.dt * frame * 5:.2f} с')
        return line, time_text
    
    anim = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
    plt.tight_layout()
    plt.show()
    
    return anim


if __name__ == "__main__":
    print(f"Скорость гравитационных волн: c = √(gH) = {np.sqrt(9.81*10):.2f} м/с")
    
    print("Запуск анимации...")
    animate_solution()
