import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


class BallisticsCalculator:
    def __init__(self, g: float = 9.81):
        self.g = g
    
    def Trajectory(self, v0: float, angle: float, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)
        
        x = vx * t
        y = vy * t - 0.5 * self.g * t**2
        
        return x, y
    
    def RangeAtAngle(self, v0: float, angle: float, target_x: float, target_y: float = 0) -> float:
        """
        Расчет разности между целевой точкой и точкой падения при данном угле

        Возвращает:
        Разность между целевой точкой и точкой падения
        """

        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)
        
        t = target_x / vx
        y = vy * t - 0.5 * self.g * t**2
        
        return y - target_y
    
    def AnalyticalSolution(self, v0: float, target_x: float, target_y: float = 0) -> List[float]:  
        sin_2alpha = target_x * self.g / v0**2
        
        if abs(sin_2alpha) > 1:
            return []  # Нет решения - цель недостижима
        
        # Два возможных угла
        alpha1 = 0.5 * np.arcsin(sin_2alpha)
        alpha2 = 0.5 * (np.pi - np.arcsin(sin_2alpha))    
        
        return [alpha1, alpha2]


class ShootingMethod:
    def __init__(self, calculator: BallisticsCalculator):
        self.calc = calculator
        self.iterations = 0
        self.convergence_history = []
    
    def BisectionMethod(self, v0: float, target_x: float, target_y: float = 0,
                        angle_min: float = 0, angle_max: float = np.pi/2,
                        tolerance: float = 1e-6, max_iter: int = 100) -> Dict:
        self.iterations = 0
        self.convergence_history = []
        
        def f(angle):
            return self.calc.RangeAtAngle(v0, angle, target_x, target_y)
        
        angle_left = angle_min
        angle_right = angle_max
        
        for i in range(max_iter):
            self.iterations = i + 1
            angle_mid = (angle_left + angle_right) / 2
            
            f_mid = f(angle_mid)
            self.convergence_history.append(abs(f_mid))
            
            if abs(f_mid) < tolerance:
                return {
                    'angle': angle_mid,
                    'error': abs(f_mid),
                    'iterations': self.iterations,
                    'convergence': self.convergence_history
                }
            
            f_left = f(angle_left)
            
            if f_left * f_mid < 0:
                angle_right = angle_mid
            else:
                angle_left = angle_mid

        return {
            'angle': angle_mid,
            'error': abs(f_mid),
            'iterations': self.iterations,
            'convergence': self.convergence_history
        }


def VisualizeResults(solve_num: Dict, v0: float, target_x: float, target_y: float = 0):
    calc = BallisticsCalculator()
    t_max = 2 * v0 / calc.g
    t = np.linspace(0, t_max, 1000)
    
    angle = solve_num['angle']
    x, y = calc.Trajectory(v0, angle, t)

    # Обрезаем траекторию при y < target_y
    mask = y >= target_y
    x, y = x[mask], y[mask]

    fig, axes = plt.subplots(1, 2, figsize=(15, 12))

    ax1 = axes[0]
    ax1.plot(x, y, color='blue', label=f"Bisection: α={np.degrees(angle):.2f}°")
    ax1.plot(target_x, target_y, 'ko', markersize=10, label='Цель')
    ax1.set_xlabel('Расстояние (м)')
    ax1.set_ylabel('Высота (м)')
    ax1.set_title('Траектория полета')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, max(target_x * 1.2, 100))
    
    ax2 = axes[1]
    ax2.semilogy(solve_num['convergence'], color='blue', marker='o', label='Bisection')
    ax2.set_xlabel('Итерация')
    ax2.set_ylabel('Ошибка (log)')
    ax2.set_title('Сходимость метода')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    v0 = 50.0         
    target_x = 200.0
    target_y = 0.0

    print(f"\nПараметры задачи:")
    print(f"  Начальная скорость: {v0} м/с")
    print(f"  Целевая точка: ({target_x}, {target_y}) м")

    calc = BallisticsCalculator()
    shooter = ShootingMethod(calc)
    
    analytical_angles = calc.AnalyticalSolution(v0, target_x, target_y)
    if analytical_angles:
        print(f"\nАналитическое решение:")
        for i, angle in enumerate(analytical_angles):
            print(f"  Угол {i+1}: {np.degrees(angle):.4f}°")
    else:
        print("\nЦель недостижима при данной начальной скорости!")
        raise
    
    solve_num = shooter.BisectionMethod(v0, target_x, target_y)

    print(f"\nЧисленное решение методом Bisection:")
    print(f"  Угол (°): {np.degrees(solve_num['angle'])}")
    print(f"  Ошибка: {solve_num['error']}")
    print(f"  Итерации: {solve_num['iterations']}")

    VisualizeResults(solve_num, v0, target_x, target_y)
