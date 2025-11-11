import numpy as np
import matplotlib.pyplot as plt

# Параметры сетки
Nx = 50  # Количество точек по горизонтали
Ny = 50  # Количество точек по вертикали

# Параметры задачи
V = 10.0  # Напряжение на пластинах
plate_width = 10  # Ширина пластин
plate_gap = 20  # Расстояние между пластинами

# Создаем сетку для потенциала, заполненную нулями
phi = np.zeros((Ny, Nx))

# Устанавливаем граничные условия Дирихле
# Боковые границы: φ = 0
phi[:, 0] = 0  # Левая граница
phi[:, -1] = 0  # Правая граница

# Верхняя и нижняя границы: φ = 0
phi[0, :] = 0  # Верхняя граница
phi[-1, :] = 0  # Нижняя граница

# Вычисляем позиции для пластин
left_plate_start = (Nx - plate_width) // 2  # Начало левой пластины
left_plate_end = left_plate_start + plate_width  # Конец левой пластины

right_plate_start = (Nx - plate_width) // 2  # Начало правой пластины
right_plate_end = right_plate_start + plate_width  # Конец правой пластины

# Позиции пластин по вертикали
top_plate_pos = (Ny - plate_gap) // 2  # Верхняя пластина
bottom_plate_pos = top_plate_pos + plate_gap  # Нижняя пластина

# Устанавливаем потенциал на пластинах
# Верхняя пластина: φ = V
phi[top_plate_pos, left_plate_start:left_plate_end] = V

# Нижняя пластина: φ = -V
phi[bottom_plate_pos, right_plate_start:right_plate_end] = -V

# Параметры метода Гаусса-Зейделя
max_iter = 1000  # Максимальное количество итераций
tolerance = 1e-4  # Точность решения

# Копируем начальные условия для сравнения
phi_old = phi.copy()

print("Начинаем решение уравнения Лапласа методом Гаусса-Зейделя...")

# Основной цикл метода Гаусса-Зейделя
for iteration in range(max_iter):
    # Проходим по всем точкам сетки (кроме границ)
    for i in range(1, Ny - 1):  # По вертикали (без верхней и нижней границ)
        for j in range(1, Nx - 1):  # По горизонтали (без левой и правой границ)

            # Пропускаем точки на пластинах (там граничные условия)
            if (i == top_plate_pos and left_plate_start <= j < left_plate_end):
                continue  # Не меняем потенциал на верхней пластине
            if (i == bottom_plate_pos and right_plate_start <= j < right_plate_end):
                continue  # Не меняем потенциал на нижней пластине

            # Формула метода Гаусса-Зейделя для уравнения Лапласа
            # ∇²φ = 0 => φ[i,j] = (φ[i+1,j] + φ[i-1,j] + φ[i,j+1] + φ[i,j-1]) / 4
            phi[i, j] = (phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1]) / 4

    # Проверяем сходимость: вычисляем максимальное изменение
    max_change = np.max(np.abs(phi - phi_old))

    # Обновляем старые значения для следующей итерации
    phi_old = phi.copy()

    # Выводим прогресс каждые 100 итераций
    if iteration % 100 == 0:
        print(f"Итерация {iteration}, максимальное изменение: {max_change:.6f}")

    # Если достигли нужной точности, выходим из цикла
    if max_change < tolerance:
        print(f"Решение сошлось на итерации {iteration}")
        break

# Вычисляем электрическое поле E = -∇φ (градиент потенциала)
# Создаем массивы для компонент электрического поля
Ex = np.zeros((Ny, Nx))  # x-компонента электрического поля
Ey = np.zeros((Ny, Nx))  # y-компонента электрического поля

# Вычисляем градиент с помощью центральных разностей
for i in range(1, Ny - 1):  # По вертикали
    for j in range(1, Nx - 1):  # По горизонтали
        # x-компонента: -dφ/dx
        Ex[i, j] = -(phi[i, j + 1] - phi[i, j - 1]) / 2
        # y-компонента: -dφ/dy
        Ey[i, j] = -(phi[i + 1, j] - phi[i - 1, j]) / 2

# Визуализация результатов
print("Создаем графики...")

# Создаем фигуру с двумя подграфиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Первый график: распределение потенциала (цветовая карта)
im = ax1.imshow(phi, cmap='jet', origin='lower', extent=[0, Nx, 0, Ny])
ax1.set_title('Распределение потенциала φ', fontsize=14)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# Добавляем цветовую шкалу для потенциала
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Потенциал φ', rotation=270, labelpad=15)

# Отмечаем положение пластин на графике
ax1.axhline(y=top_plate_pos, xmin=left_plate_start / Nx, xmax=left_plate_end / Nx,
            color='white', linewidth=3, label='Пластина +V')
ax1.axhline(y=bottom_plate_pos, xmin=right_plate_start / Nx, xmax=right_plate_end / Nx,
            color='black', linewidth=3, label='Пластина -V')
ax1.legend()

# Второй график: линии напряженности электрического поля
# Создаем сетку координат для векторного поля
X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))

# Рисуем векторное поле (каждый 3-й вектор для лучшей читаемости)
ax2.quiver(X[::3, ::3], Y[::3, ::3], Ex[::3, ::3], Ey[::3, ::3],
           scale=50, color='red', alpha=0.7)

# Добавляем контурные линии потенциала
contour = ax2.contour(X, Y, phi, levels=15, colors='blue', alpha=0.7)
ax2.clabel(contour, inline=True, fontsize=8)

ax2.set_title('Электрическое поле E = -∇φ', fontsize=14)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

# Отмечаем положение пластин на втором графике
ax2.axhline(y=top_plate_pos, xmin=left_plate_start / Nx, xmax=left_plate_end / Nx,
            color='green', linewidth=3, label='Пластина +V')
ax2.axhline(y=bottom_plate_pos, xmin=right_plate_start / Nx, xmax=right_plate_end / Nx,
            color='orange', linewidth=3, label='Пластина -V')
ax2.legend()

plt.tight_layout()
plt.show()

# Дополнительная визуализация: 3D поверхность потенциала
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Создаем сетку для 3D графика
X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))

# Рисуем 3D поверхность
surf = ax.plot_surface(X, Y, phi, cmap='viridis', alpha=0.8)

ax.set_title('3D распределение потенциала', fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Потенциал φ')

plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()
