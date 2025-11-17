import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Параметры сетки
Nx, Ny = 100, 100  # Размеры сетки
Lx, Ly = 1.0, 1.0  # Физические размеры области
dx, dy = Lx/(Nx-1), Ly/(Ny-1)  # Шаги сетки

# Параметры задачи
V = 1.0  # Напряжение на пластинах
plate_width = 0.2  # Ширина пластин
plate_pos_y1 = 0.3  # Положение первой пластины
plate_pos_y2 = 0.7  # Положение второй пластины

# Инициализация потенциала
phi = np.zeros((Ny, Nx))

# Установка граничных условий
# Боковые границы: phi = 0
phi[:, 0] = 0.0    # Левая граница
phi[:, -1] = 0.0   # Правая граница

# Пластины
plate_start_x = int((Lx - plate_width)/2 / dx)
plate_end_x = int((Lx + plate_width)/2 / dx)
plate_y1 = int(plate_pos_y1 / dy)
plate_y2 = int(plate_pos_y2 / dy)

# Верхняя пластина: phi = -V
phi[plate_y2, plate_start_x:plate_end_x] = -V

# Нижняя пластина: phi = V
phi[plate_y1, plate_start_x:plate_end_x] = V

# Копия для граничных условий (чтобы не перезаписывать их)
phi_boundary = phi.copy()


def gauss_seidel(phi, phi_boundary, max_iter=10000, tolerance=1e-6):
    """
    Решение уравнения Лапласа методом Гаусса-Зейделя
    """
    for iteration in range(max_iter):
        phi_old = phi.copy()
        
        # Обновление внутренних точек
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                # Пропускаем точки на пластинах (граничные условия Дирихле)
                if phi_boundary[i, j] != 0:
                    continue
                
                # Метод Гаусса-Зейделя
                phi[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + 
                                   phi[i, j+1] + phi[i, j-1])
        
        # Восстанавливаем граничные условия
        phi[phi_boundary != 0] = phi_boundary[phi_boundary != 0]
        
        # Проверка сходимости
        max_diff = np.max(np.abs(phi - phi_old))
        if max_diff < tolerance:
            print(f"Сходимость достигнута на итерации {iteration+1}")
            break
    
    return phi


def calculate_electric_field(phi, dx, dy):
    """
    Вычисление напряженности электрического поля E = -∇φ
    """
    # Вычисление градиента
    Ey, Ex = np.gradient(phi, dy, dx)
    
    # E = -∇φ
    Ex = -Ex
    Ey = -Ey
    
    return Ex, Ey


if __name__ == "__main__":
    print("Решение уравнения Лапласа методом Гаусса-Зейделя...")
    phi = gauss_seidel(phi, phi_boundary)

    print("Вычисление электрического поля...")
    Ex, Ey = calculate_electric_field(phi, dx, dy)

    # Создание координатной сетки для визуализации
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Поле потенциала (цветовая карта)
    im = ax1.contourf(X, Y, phi, levels=50, cmap=cm.jet)
    ax1.set_title('Распределение потенциала φ(x,y)', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im, ax=ax1, label='Потенциал φ')

    # Отметка положения пластин
    ax1.axhline(y=plate_pos_y1, xmin=(0.5 - plate_width/2), xmax=(0.5 + plate_width/2), 
            color='white', linewidth=3, label='Пластина (+V)')
    ax1.axhline(y=plate_pos_y2, xmin=(0.5 - plate_width/2), xmax=(0.5 + plate_width/2), 
            color='black', linewidth=3, label='Пластина (-V)')
    ax1.legend()

    # 2. Линии напряженности электрического поля
    # Уменьшаем плотность векторов для лучшей читаемости
    step = 4
    ax2.quiver(X[::step, ::step], Y[::step, ::step], 
            Ex[::step, ::step], Ey[::step, ::step], 
            scale=20, color='red', alpha=0.7)
    ax2.set_title('Векторное поле напряженности E = -∇φ', fontsize=14)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Добавляем эквипотенциальные линии
    ax2.contour(X, Y, phi, levels=15, colors='blue', alpha=0.5, linewidths=0.5)

    # Отметка положения пластин
    ax2.axhline(y=plate_pos_y1, xmin=(0.5 - plate_width/2), xmax=(0.5 + plate_width/2), 
            color='green', linewidth=3, label='Пластина (+V)')
    ax2.axhline(y=plate_pos_y2, xmin=(0.5 - plate_width/2), xmax=(0.5 + plate_width/2), 
            color='purple', linewidth=3, label='Пластина (-V)')
    ax2.legend()

    plt.tight_layout()
    plt.show()
