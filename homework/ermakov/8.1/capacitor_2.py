import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time


def setup_problem(Nx, Ny, V, square_params):
    """
    Инициализация задачи с проводящим квадратом
    """
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx/(Nx-1), Ly/(Ny-1)
    
    # Инициализация потенциала
    phi = np.zeros((Ny, Nx))
    
    # Граничные условия для пластин конденсатора
    plate_width = 0.6
    plate_pos_y1 = 0.2
    plate_pos_y2 = 0.8
    
    plate_start_x = int((Lx - plate_width)/2 / dx)
    plate_end_x = int((Lx + plate_width)/2 / dx)
    plate_y1 = int(plate_pos_y1 / dy)
    plate_y2 = int(plate_pos_y2 / dy)
    
    # Верхняя пластина: phi = -V
    phi[plate_y2, plate_start_x:plate_end_x] = -V
    # Нижняя пластина: phi = V
    phi[plate_y1, plate_start_x:plate_end_x] = V
    
    # Проводящий квадрат
    square_x, square_y, square_size = square_params
    square_start_x = int(square_x / dx)
    square_end_x = int((square_x + square_size) / dx)
    square_start_y = int(square_y / dy)
    square_end_y = int((square_y + square_size) / dy)
    
    # Создаем маску для квадрата
    square_mask = np.zeros((Ny, Nx), dtype=bool)
    square_mask[square_start_y:square_end_y, square_start_x:square_end_x] = True
    
    # Потенциал на квадрате (незаряженный проводник)
    phi[square_mask] = 0.0
    
    # Копия для граничных условий
    phi_boundary = phi.copy()
    
    return phi, phi_boundary, square_mask, dx, dy


def jacobi_method(phi, phi_boundary, square_mask, max_iter=10000, tolerance=1e-6):
    """
    Метод Якоби для решения уравнения Лапласа
    """
    phi_new = phi.copy()
    errors = []
    
    for iteration in range(max_iter):
        phi_old = phi_new.copy()
        
        # Обновление всех точек
        for i in range(1, phi.shape[0]-1):
            for j in range(1, phi.shape[1]-1):
                # Пропускаем граничные точки и квадрат
                if phi_boundary[i, j] != 0 or square_mask[i, j]:
                    continue
                
                # Метод Якоби
                phi_new[i, j] = 0.25 * (phi_old[i+1, j] + phi_old[i-1, j] + 
                                       phi_old[i, j+1] + phi_old[i, j-1])
        
        # Для проводящего квадрата: среднее значение потенциала на границе
        update_conductor_potential(phi_new, square_mask)
        
        # Восстанавливаем фиксированные граничные условия
        mask_fixed = (phi_boundary != 0) & ~square_mask
        phi_new[mask_fixed] = phi_boundary[mask_fixed]
        
        # Проверка сходимости
        error = np.max(np.abs(phi_new - phi_old))
        errors.append(error)
        
        if error < tolerance:
            print(f"Якоби: сходимость достигнута на итерации {iteration+1}")
            break
    
    return phi_new, errors


def gauss_seidel_method(phi, phi_boundary, square_mask, max_iter=10000, tolerance=1e-6):
    """
    Метод Гаусса-Зейделя для решения уравнения Лапласа
    """
    phi_work = phi.copy()
    errors = []
    
    for iteration in range(max_iter):
        phi_old = phi_work.copy()
        
        # Обновление внутренних точек
        for i in range(1, phi.shape[0]-1):
            for j in range(1, phi.shape[1]-1):
                # Пропускаем граничные точки и квадрат
                if phi_boundary[i, j] != 0 or square_mask[i, j]:
                    continue
                
                # Метод Гаусса-Зейделя
                phi_work[i, j] = 0.25 * (phi_work[i+1, j] + phi_work[i-1, j] + 
                                        phi_work[i, j+1] + phi_work[i, j-1])
        
        # Для проводящего квадрата
        update_conductor_potential(phi_work, square_mask)
        
        # Восстанавливаем фиксированные граничные условия
        mask_fixed = (phi_boundary != 0) & ~square_mask
        phi_work[mask_fixed] = phi_boundary[mask_fixed]
        
        # Проверка сходимости
        error = np.max(np.abs(phi_work - phi_old))
        errors.append(error)
        
        if error < tolerance:
            print(f"Гаусс-Зейдель: сходимость достигнута на итерации {iteration+1}")
            break
    
    return phi_work, errors


def update_conductor_potential(phi, square_mask):
    """
    Обновление потенциала проводящего квадрата (усреднение граничных значений)
    """
    # Находим границу квадрата
    from scipy import ndimage
    
    # Создаем маску для границы квадрата
    erosion_mask = ndimage.binary_erosion(square_mask)
    border_mask = square_mask & ~erosion_mask
    
    # Вычисляем средний потенциал вокруг квадрата
    border_values = []
    for i in range(1, phi.shape[0]-1):
        for j in range(1, phi.shape[1]-1):
            if border_mask[i, j]:
                # Собираем значения вокруг границы
                neighbors = []
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < phi.shape[0] and 0 <= nj < phi.shape[1]:
                        if not square_mask[ni, nj]:
                            neighbors.append(phi[ni, nj])
                
                if neighbors:
                    border_values.extend(neighbors)
    
    if border_values:
        avg_potential = np.mean(border_values)
        phi[square_mask] = avg_potential


def calculate_electric_field(phi, dx, dy):
    """
    Вычисление напряженности электрического поля E = -∇φ
    """
    Ey, Ex = np.gradient(phi, dy, dx)
    Ex = -Ex
    Ey = -Ey
    return Ex, Ey


def visualize_results(phi, square_mask, dx, dy, method_name):
    """
    Визуализация результатов для одного метода
    """
    Ex, Ey = calculate_electric_field(phi, dx, dy)
    
    # Создание координатной сетки
    x = np.linspace(0, 1.0, phi.shape[1])
    y = np.linspace(0, 1.0, phi.shape[0])
    X, Y = np.meshgrid(x, y)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Поле потенциала
    im = ax1.contourf(X, Y, phi, levels=50, cmap=cm.jet)
    ax1.contour(X, Y, phi, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    ax1.set_title(f'Потенциал φ ({method_name})', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im, ax=ax1, label='Потенциал φ')
    
    # Отметка квадрата
    square_indices = np.where(square_mask)
    if len(square_indices[0]) > 0:
        ax1.fill_between(x[square_indices[1].min():square_indices[1].max()+1],
                        y[square_indices[0].min()], y[square_indices[0].max()],
                        color='gray', alpha=0.7, label='Проводящий квадрат')
    
    # 2. Силовые линии
    # Уменьшаем плотность векторов
    step = 3
    magnitude = np.sqrt(Ex[::step, ::step]**2 + Ey[::step, ::step]**2)
    magnitude[magnitude == 0] = 1  # Избегаем деления на ноль
    
    ax2.quiver(X[::step, ::step], Y[::step, ::step], 
              Ex[::step, ::step]/magnitude, Ey[::step, ::step]/magnitude,
              scale=30, color='red', alpha=0.7, width=0.005)
    
    # Линии тока (перпендикулярно эквипотенциальным)
    ax2.streamplot(X, Y, Ex, Ey, color='blue', density=1.5, linewidth=1, 
                  arrowstyle='->', arrowsize=1.0)
    
    ax2.set_title(f'Силовые линии ({method_name})', fontsize=14)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Отметка квадрата
    if len(square_indices[0]) > 0:
        ax2.fill_between(x[square_indices[1].min():square_indices[1].max()+1],
                        y[square_indices[0].min()], y[square_indices[0].max()],
                        color='gray', alpha=0.7, label='Проводящий квадрат')
    
    plt.tight_layout()
    plt.show()
    
    return Ex, Ey


if __name__ == "__main__":
    Nx, Ny = 100, 100
    V = 1.0
    square_params = (0.35, 0.35, 0.3)  # (x, y, size)
    
    print("Инициализация задачи...")
    phi_init, phi_boundary, square_mask, dx, dy = setup_problem(Nx, Ny, V, square_params)
    
    # Решение методом Якоби
    print("\n=== Решение методом Якоби ===")
    start_time = time.time()
    phi_jacobi, errors_jacobi = jacobi_method(phi_init.copy(), phi_boundary, square_mask)
    time_jacobi = time.time() - start_time
    print(f"Время выполнения: {time_jacobi:.2f} сек")
    
    # Решение методом Гаусса-Зейделя
    print("\n=== Решение методом Гаусса-Зейделя ===")
    start_time = time.time()
    phi_gs, errors_gs = gauss_seidel_method(phi_init.copy(), phi_boundary, square_mask)
    time_gs = time.time() - start_time
    print(f"Время выполнения: {time_gs:.2f} сек")
    
    # Визуализация результатов
    print("\nВизуализация результатов...")
    Ex_j, Ey_j = visualize_results(phi_jacobi, square_mask, dx, dy, "Якоби")
    Ex_gs, Ey_gs = visualize_results(phi_gs, square_mask, dx, dy, "Гаусс-Зейдель")
    
    # Сравнение сходимости
    plt.figure(figsize=(10, 6))
    plt.semilogy(errors_jacobi, 'b-', label='Метод Якоби', alpha=0.7)
    plt.semilogy(errors_gs, 'r-', label='Метод Гаусса-Зейделя', alpha=0.7)
    plt.xlabel('Номер итерации')
    plt.ylabel('Максимальная ошибка')
    plt.title('Сравнение скорости сходимости методов')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Сравнение результатов
    diff = np.max(np.abs(phi_jacobi - phi_gs))
    print(f"\n=== Сравнение методов ===")
    print(f"Максимальное расхождение решений: {diff:.2e}")
    print(f"Отношение времени выполнения (Якоби/Г-З): {time_jacobi/time_gs:.2f}")
    print(f"Отношение числа итераций: {len(errors_jacobi)}/{len(errors_gs)} = {len(errors_jacobi)/len(errors_gs):.2f}")
    
    # Потенциал на квадрате
    square_potential_j = np.mean(phi_jacobi[square_mask])
    square_potential_gs = np.mean(phi_gs[square_mask])
    print(f"Потенциал квадрата (Якоби): {square_potential_j:.4f}")
    print(f"Потенциал квадрата (Г-З): {square_potential_gs:.4f}")
