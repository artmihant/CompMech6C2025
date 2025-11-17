# -*- coding: utf-8 -*-
"""
Реализация модели реакции-диффузии Грэя–Скотта на GPU с использованием Numba.
Код предназначен для запуска в среде Kaggle Notebooks.

Модель описывает изменение концентраций двух химических веществ u и v:
∂u/∂t = Du * ∇²u - u*v² + f*(1 - u)
∂v/∂t = Dv * ∇²v + u*v² - (f + k)*v

Здесь:
- Du, Dv: коэффициенты диффузии
- f: скорость подпитки (feed rate)
- k: скорость утилизации (kill rate)
- ∇²: оператор Лапласа (описывает диффузию)

Этот код демонстрирует:
1.  Решение системы УЧП на GPU.
2.  Использование 2D-сетки нитей для обработки 2D-массива.
3.  Применение периодических граничных условий.
4.  Технику двойной буферизации для итерационного процесса.
5.  Создание анимации процесса с помощью Matplotlib.
"""
import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- 1. Параметры симуляции ---

# Размер сетки
N = 512

# Параметры модели Грэя–Скотта. Разные значения дают разные паттерны.
# Этот набор ("Solitons") дает стабильные "клетки"
f = 0.04
k = 0.06
Du = 0.16
Dv = 0.08
dt = 1.0  # Шаг по времени (дискретный)

# Параметры анимации
NUM_FRAMES = 200  # Количество кадров в итоговой анимации
STEPS_PER_FRAME = 50  # Сколько шагов расчета делать между кадрами анимации

# --- 2. GPU Kernel ---

@cuda.jit
def update_grid_kernel(u_in, v_in, u_out, v_out, Du, Dv, f, k, dt, N):
    """
    GPU-ядро для выполнения одного шага симуляции.
    Не использует shared memory, так как для 5-точечного шаблона
    выигрыш будет минимален из-за эффективного кэширования в L1/L2.
    """
    # Получаем глобальные координаты нити в 2D-сетке
    j, i = cuda.grid(2)

    # Убедимся, что мы не выходим за границы массива
    if i >= N or j >= N:
        return

    # --- Вычисление Лапласиана с периодическими граничными условиями ---
    # Периодические условия означают, что сетка "зациклена":
    # правый край соединен с левым, а верхний - с нижним.
    # Оператор % (остаток от деления) элегантно реализует это.
    u_center = u_in[j, i]
    v_center = v_in[j, i]

    u_up = u_in[(j - 1 + N) % N, i]
    u_down = u_in[(j + 1) % N, i]
    u_left = u_in[j, (i - 1 + N) % N]
    u_right = u_in[j, (i + 1) % N]

    v_up = v_in[(j - 1 + N) % N, i]
    v_down = v_in[(j + 1) % N, i]
    v_left = v_in[j, (i - 1 + N) % N]
    v_right = v_in[j, (i + 1) % N]
    
    # 5-точечный шаблон для дискретного Лапласиана
    laplacian_u = u_up + u_down + u_left + u_right - 4 * u_center
    laplacian_v = v_up + v_down + v_left + v_right - 4 * v_center

    # --- Вычисление реакционного члена ---
    reaction_term = u_center * v_center * v_center

    # --- Полная формула (явная схема Эйлера) ---
    # Вычисляем новое значение u
    u_new = u_center + (Du * laplacian_u - reaction_term + f * (1 - u_center)) * dt
    # Вычисляем новое значение v
    v_new = v_center + (Dv * laplacian_v + reaction_term - (f + k) * v_center) * dt
    
    # Записываем результат в выходные массивы
    u_out[j, i] = u_new
    v_out[j, i] = v_new


# --- 3. Подготовка данных и запуск ---

# Проверка, что GPU доступен
print(f"GPU: {cuda.get_current_device().name.decode('UTF-8')}\n")

# --- Начальные условия ---
# u - почти везде 1, v - почти везде 0
u_host = np.ones((N, N), dtype=np.float64)
v_host = np.zeros((N, N), dtype=np.float64)

# Создаем "посев": небольшой случайный квадрат в центре
r = 32 # радиус квадрата
center = N // 2
u_host[center-r:center+r, center-r:center+r] = 0.50
v_host[center-r:center+r, center-r:center+r] = 0.25

# Добавляем шум, чтобы нарушить симметрию и запустить паттернообразование
u_host += 0.1 * np.random.random((N, N))
v_host += 0.1 * np.random.random((N, N))

# --- Копирование данных на GPU ---
# Это "входные" буферы
u_device = cuda.to_device(u_host)
v_device = cuda.to_device(v_host)
# Это "выходные" буферы для результатов шага
u_new_device = cuda.device_array_like(u_device)
v_new_device = cuda.device_array_like(v_device)

# --- Настройка сетки GPU ---
# Обычно 16x16 или 32x32 нити на блок - хороший выбор
threads_per_block = (16, 16)
# Рассчитываем, сколько блоков нужно, чтобы покрыть всю сетку
blocks_per_grid_x = math.ceil(N / threads_per_block[0])
blocks_per_grid_y = math.ceil(N / threads_per_block[1])
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# --- 4. Настройка анимации ---

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xticks([])
ax.set_yticks([])
# Мы будем анимировать концентрацию v, так как она обычно образует более четкие паттерны
# vmin и vmax фиксируют цветовую шкалу для консистентности
img = ax.imshow(v_host, cmap='magma', vmin=0, vmax=0.5)
total_steps = 0
title = ax.set_title(f"Шаг: {total_steps}")


def animate(frame):
    """Функция, вызываемая для каждого кадра анимации."""
    global u_device, v_device, u_new_device, v_new_device, total_steps

    # Запускаем расчет на GPU на STEPS_PER_FRAME шагов
    for _ in range(STEPS_PER_FRAME):
        # Вызов ядра
        update_grid_kernel[blocks_per_grid, threads_per_block](
            u_device, v_device, u_new_device, v_new_device,
            Du, Dv, f, k, dt, N
        )
        # Техника двойной буферизации:
        # результат предыдущего шага становится входом для следующего.
        # Это позволяет избежать копирования данных и просто "переключить" указатели.
        u_device, u_new_device = u_new_device, u_device
        v_device, v_new_device = v_new_device, v_device

    # Обновляем счетчик шагов
    total_steps += STEPS_PER_FRAME
    
    # Копируем результат с GPU на CPU для отображения
    v_result_host = v_device.copy_to_host()
    
    # Обновляем данные на графике
    img.set_array(v_result_host)
    title.set_text(f"Шаг: {total_steps}")
    
    # Печатаем прогресс в консоль
    # print(f"Кадр {frame+1}/{NUM_FRAMES}, Шаг {total_steps}")

    return [img, title]

# Создаем и сохраняем анимацию
anim = FuncAnimation(
    fig,
    animate,
    frames=NUM_FRAMES,
    interval=30,  # задержка между кадрами в мс
    blit=True     # blitting ускоряет отрисовку
)

# Сохраняем в GIF. PillowWriter нужен для этого.
from IPython.display import HTML
HTML(anim.to_html5_video())
