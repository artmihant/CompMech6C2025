import time
import matplotlib.pyplot as plt, matplotlib.animation
import numpy as np
from numba import cuda

""" 
Метод решеточных уравнений Больцмана (Lattice Boltzmann Method, LBM)
для моделирования течения вязкой несжимаемой жидкости в 2D

Основная идея:
- Дискретное пространство скоростей (9 направлений в модели D2Q9)
- Функции распределения f_i(x,t) описывают плотность частиц с заданной скоростью
- Два шага: Streaming (перенос частиц) + Collision (релаксация к равновесию)
- Макроскопические величины (плотность, скорость) — моменты распределения
"""

""" Зададим константы и свойства задачи """

NUM_FRAMES = 200  # Количество кадров в итоговой анимации
STEPS_PER_FRAME = 50  # Сколько шагов расчета делать между кадрами анимации

Viscosity = 0.01                                # кинематическая вязкость жидкости, м²/с
Height, Width = 80, 200                         # размеры решетки (узлов по y и x)

BarrierCenter = Height//2, Height//2            # центр круглого препятствия
BarrierRadius = Height//10                      # радиус препятствия

U0 = np.array([0.1, 0])                        # начальная скорость потока (в долях скорости звука)

# Инициализация макроскопических полей (текущее состояние)
Ux  = np.zeros((Height, Width)) + U0[0]        # поле скорости по x
Uy  = np.zeros((Height, Width)) + U0[1]        # поле скорости по y
Rho = np.ones((Height, Width))                 # поле плотности (безразмерная)

# Сохранение начальных условий (не используется в текущей версии)
Ux0  = np.zeros((Height, Width)) + U0[0]
Uy0  = np.zeros((Height, Width)) + U0[1]
Rho0 = np.ones((Height, Width))


def BarrierShape():
    """ 
    Инициализируем форму твердого препятствия (барьера).
    
    Returns:
        barrier: булева матрица, где True обозначает твердые узлы
    """
    barrier = np.zeros((Height, Width), bool)

    # Создаем круглое препятствие в центре
    for y in range(barrier.shape[0]):
        for x in range(barrier.shape[1]):
            if (x - BarrierCenter[0])**2 + (y - BarrierCenter[1])**2 < (BarrierRadius)**2:
                barrier[y,x] = True
    
    # Опционально: можно добавить хвост за кругом (закомментировано)
    # barrier[(Height//2), ((Height//2)):((Height//2)+4*(Height//10))] = True

    return barrier


""" 
Зададим свойства шаблона решетки D2Q9 

D2Q9 означает: 2D пространство, 9 дискретных скоростей
Нумерация направлений (q):
    6 -- 7 -- 8       NW -- N -- NE
    |    |    |        |    |    |
    3 -- 4 -- 5   ==>  W -- C -- E
    |    |    |        |    |    |
    0 -- 1 -- 2       SW -- S -- SE
"""

D = 2  # Размерность модели (2D)
Q = 9  # Число дискретных скоростей в шаблоне

# Массив дискретных скоростей: V[q] = (vx, vy)
# Каждая строка — вектор скорости для направления q
V = np.array([
    [-1, 1],[ 0, 1],[ 1, 1],   # SW, S, SE (нижний ряд -> на самом деле это верхний в индексации)
    [-1, 0],[ 0, 0],[ 1, 0],   # W, C, E (центральный ряд)
    [-1,-1],[ 0,-1],[ 1,-1]    # NW, N, NE (верхний ряд -> на самом деле это нижний в индексации)
])

#
# Весовые коэффициенты для равновесного распределения
# Зависят от нормы вектора скорости: |v|=0 -> 4/9, |v|=1 -> 1/9, |v|=√2 -> 1/36
W = np.array([
    1/36, 1/9, 1/36,   # диагональные направления (|v|=√2)
    1/9,  4/9, 1/9,    # ортогональные + покой (|v|=0,1)
    1/36, 1/9, 1/36    # диагональные направления (|v|=√2)
])

# Массив индексов противоположных направлений для bounce-back
# 0 <-> 8, 1 <-> 7, 2 <-> 6, 3 <-> 5, 4 <-> 4
Opposite = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int32)

C = 1/3**0.5  # Скорость звука в решеточных единицах: c_s = 1/√3 ≈ 0.577


# Переносим константы на устройство (GPU)
V_device = cuda.to_device(V)
W_device = cuda.to_device(W)
Opposite_device = cuda.to_device(Opposite)


@cuda.jit
def lbm_stream_bounce_kernel(f_in, f_out, barrierC, V_dev, Opp_dev, height, width):
    """
    CUDA-ядро для шага STREAMING + BOUNCE-BACK.

    Используем схему «pull»: каждая нить вычисляет новые значения
    для одной ячейки (y, x) во всех направлениях q.
    """
    y, x = cuda.grid(2)
    if y >= height or x >= width:
        return

    # Твердый узел: все распределения здесь нам не нужны
    if barrierC[y, x]:
        for q in range(Q):
            f_out[q, y, x] = 0.0
        return

    for q in range(Q):
        vx = V_dev[q, 0]
        vy = V_dev[q, 1]

        # Координаты узла, из которого "приходит" частица
        ys = y - vy
        xs = x - vx

        if (ys >= 0) and (ys < height) and (xs >= 0) and (xs < width):
            # Если соседний узел — твердое препятствие, реализуем half-way bounce-back:
            # поток, который должен был прийти из твердого узла,
            # заменяем на отраженный из противоположного направления в том же узле.
            if barrierC[ys, xs]:
                qo = Opp_dev[q]
                f_out[q, y, x] = f_in[qo, y, x]
            else:
                f_out[q, y, x] = f_in[q, ys, xs]
        # Если вышли за пределы области — значение будет позже
        # переопределено граничными условиями.


@cuda.jit
def lbm_collision_kernel(f, tau, V_dev, W_dev, height, width):
    """
    CUDA-ядро для шага столкновения (collision) в методе LBM.
    Каждая нить обрабатывает одну ячейку решетки (y, x).
    """
    y, x = cuda.grid(2)
    if y >= height or x >= width:
        return

    rho = 0.0
    ux = 0.0
    uy = 0.0

    # Вычисляем плотность и импульс в узле
    for q in range(Q):
        fi = f[q, y, x]
        rho += fi
        ux += fi * V_dev[q, 0]
        uy += fi * V_dev[q, 1]

    # Защита от деления на ноль (на всякий случай)
    if rho <= 0.0:
        return

    ux /= rho
    uy /= rho

    u2 = (ux * ux + uy * uy) / (C * C)

    # Релаксация к равновесному распределению
    for q in range(Q):
        vx = V_dev[q, 0]
        vy = V_dev[q, 1]
        uv = (vx * ux + vy * uy) / (C * C)
        feq = rho * W_dev[q] * (1.0 + uv + 0.5 * uv * uv - 0.5 * u2)
        f[q, y, x] += (feq - f[q, y, x]) / tau


@cuda.jit
def lbm_boundary_kernel(f, f_out, height, width):
    """
    CUDA-ядро для наложения граничных условий на входе/выходе области.
    """
    y, x = cuda.grid(2)
    if y >= height or x >= width:
        return

    for q in range(Q):
        # левая граница (вход потока)
        if x == 0:
            f[q, y, 0] = f_out[q, y, 0]
        # правая граница (выход потока)
        if x == width - 1:
            f[q, y, width - 1] = f_out[q, y, width - 1]
        # нижняя граница
        if y == 0:
            f[q, 0, x] = f_out[q, 0, x]
        # верхняя граница
        if y == height - 1:
            f[q, height - 1, x] = f_out[q, height - 1, x]


def InitBarrier():
    """ 
    Создаем маски для граничных условий на барьере (bounce-back).
    
    Для каждого направления q создаем маску узлов-жидкости, которые 
    граничат с твердым барьером в направлении q.
    
    Физический смысл: когда частица попадает в твердую стенку,
    она отражается обратно (bounce-back boundary condition).
    
    Returns:
        массив из 9 масок (по одной на каждое направление D2Q9)
    """
    barrierC = BarrierShape()  # Центральная маска (сам барьер)

    # Создаем смещенные маски для каждого направления
    # np.roll сдвигает массив циклически
    barrierN = np.roll(barrierC,  1, axis=0)   # North: сдвиг вверх по y
    barrierS = np.roll(barrierC, -1, axis=0)   # South: сдвиг вниз по y
    barrierE = np.roll(barrierC,  1, axis=1)   # East: сдвиг вправо по x
    barrierW = np.roll(barrierC, -1, axis=1)   # West: сдвиг влево по x
    barrierNE = np.roll(barrierN,  1, axis=1)  # NorthEast: диагональ
    barrierNW = np.roll(barrierN, -1, axis=1)  # NorthWest: диагональ
    barrierSE = np.roll(barrierS,  1, axis=1)  # SouthEast: диагональ
    barrierSW = np.roll(barrierS, -1, axis=1)  # SouthWest: диагональ

    # Возвращаем массив масок в порядке, соответствующем нумерации направлений
    return np.array([
        barrierNW, barrierN, barrierNE,
        barrierW,  barrierC, barrierE,
        barrierSW, barrierS, barrierSE
    ])


def F_stat(Ux, Uy, Rho):
    """ 
    Вычисляем равновесное распределение Максвелла-Больцмана (дискретное).
    
    Это целевое состояние, к которому релаксирует система во время столкновений.
    
    Формула (разложение Максвелла-Больцмана до 2-го порядка по скорости):
        f_i^eq = ρ * w_i * [1 + (v_i·u)/c_s^2 + (v_i·u)^2/(2c_s^4) - u^2/(2c_s^2)]
    
    где:
        ρ - плотность
        w_i - весовой коэффициент направления i
        v_i - дискретная скорость направления i
        u - макроскопическая скорость
        c_s - скорость звука
    
    Args:
        Ux, Uy: компоненты макроскопической скорости
        Rho: макроскопическая плотность
        
    Returns:
        f_stat: равновесное распределение для всех 9 направлений
    """
    # Вычисляем скалярное произведение (v_i · u) для каждого направления
    UV = np.zeros((Q, Height, Width)) 
    for q in range(Q):
        UV[q] = (V[q,0]*Ux + V[q,1]*Uy)/C**2

    # Квадрат макроскопической скорости (нормированный)
    U2 = (Ux**2 + Uy**2)/C**2

    # Вычисляем равновесное распределение для всех направлений
    f_stat = np.zeros((Q, Height, Width))
    for q in range(Q):
        f_stat[q] = Rho * W[q] * (1 + UV[q] + UV[q]**2/2 - U2/2)

    return f_stat

def Mode0(f):
    """ 
    Вычисляем 0-й момент распределения — плотность.
    
    Формула: ρ = Σ_i f_i
    
    Физический смысл: суммарная плотность частиц во всех направлениях.
    
    Args:
        f: функция распределения (массив размера [Q, Height, Width])
        
    Returns:
        mode: поле плотности (массив размера [Height, Width])
    """
    mode = np.zeros((Height, Width))
    for q in range(Q):
        mode += f[q]
    return mode

def Mode1(f):
    """ 
    Вычисляем 1-й момент распределения — импульс (плотность × скорость).
    
    Формула: ρu_α = Σ_i f_i * v_{i,α}
    
    Физический смысл: суммарный импульс всех частиц в каждом направлении (x, y).
    
    Args:
        f: функция распределения (массив размера [Q, Height, Width])
        
    Returns:
        mode: поле импульса (массив размера [D, Height, Width])
              mode[0] = ρ * Ux, mode[1] = ρ * Uy
    """
    mode = np.zeros((D, Height, Width))
    for q in range(Q):
        for d1 in range(D):
            mode[d1] += f[q]*V[q,d1]
    return mode

def Mode2(f):
    """ 
    Вычисляем 2-й момент распределения — тензор импульсного потока.
    
    Формула: Π_{αβ} = Σ_i f_i * v_{i,α} * v_{i,β}
    
    Физический смысл: связан с тензором напряжений и давлением в жидкости.
    Из этого можно извлечь информацию о вязких напряжениях.
    
    Args:
        f: функция распределения (массив размера [Q, Height, Width])
        
    Returns:
        mode: тензор 2-го ранга (массив размера [D, D, Height, Width])
    """
    mode = np.zeros((D,D,Height, Width))
    for q in range(Q):
        for d1 in range(D):
            for d2 in range(D):
                mode[d1,d2] += f[q]*V[q,d1]*V[q,d2]
    return mode


def iter(f, f_out, barrier):
    """
    CPU-реализация шага LBM оставлена для справки (не используется в GPU-анимации).
    """
    raise NotImplementedError("В GPU-версии используем только CUDA-ядра, функция iter не вызывается.")

def curl(ux, uy):
    """ 
    Вычисляем завихренность (ротор, vorticity) поля скорости.
    
    Завихренность — это мера вращательного движения жидкости.
    В 2D: ω = ∂u_y/∂x - ∂u_x/∂y
    
    Используем конечные разности второго порядка точности.
    
    Args:
        ux, uy: компоненты поля скорости
        
    Returns:
        завихренность (скалярное поле)
    """
    return np.roll(uy,-1,axis=1) - np.roll(uy,1,axis=1) - np.roll(ux,-1,axis=0) + np.roll(ux,1,axis=0)

"""
Главная функция: инициализация и запуск симуляции с визуализацией.
"""

# Инициализация геометрии барьера
barrier = InitBarrier()

# Инициализация функции распределения из равновесного состояния (на хосте)
F = F_stat(Ux, Uy, Rho)

# Фиксированное распределение для граничных условий (постоянный поток, хост)
F_out = F_stat(Ux, Uy, Rho)

# Маска твердого препятствия (центральная)
barrierC = barrier[4]

# --- Подготовка данных на устройстве (GPU) ---
F_device = cuda.to_device(F)
F_tmp_device = cuda.device_array_like(F)
F_out_device = cuda.to_device(F_out)
barrierC_device = cuda.to_device(barrierC)

# Настройки сетки CUDA: по одному потоку на узел решетки
threadsperblock = (16, 16)
blockspergrid_y = (Height + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_x = (Width + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_y, blockspergrid_x)

# Время релаксации (постоянное)
tau = 0.5 + Viscosity/C**2

# ========== НАСТРОЙКА ВИЗУАЛИЗАЦИИ ==========
fig, ax = plt.subplots()

# Визуализируем завихренность (показывает вихри)
# Используем цветовую карту jet с нормировкой от -0.1 до 0.1
fluidImage = ax.imshow(curl(Ux, Uy), origin='lower', norm=plt.Normalize(-.1,.1), 
                                    cmap=plt.get_cmap('jet'), interpolation='none')

# Визуализируем барьер полупрозрачным серым цветом
bImageArray = np.zeros((Height, Width, 4), np.uint8)
bImageArray[barrier[4],3] = 100  # альфа-канал (прозрачность) для барьера
barrierImage = plt.imshow(bImageArray, origin='lower', interpolation='none')


def nextFrame(_):
    """
    Функция обновления кадра анимации.
    Вызывается автоматически для каждого нового кадра.
    """
    global F_device, F_tmp_device

    # Измеряем время вычисления одного кадра (STEPS_PER_FRAME итераций) на GPU
    t0 = time.time()

    for _ in range(STEPS_PER_FRAME):
        # STREAMING + BOUNCE-BACK
        lbm_stream_bounce_kernel[blockspergrid, threadsperblock](
            F_device, F_tmp_device, barrierC_device, V_device, Opposite_device,
            Height, Width
        )

        # COLLISION
        lbm_collision_kernel[blockspergrid, threadsperblock](
            F_tmp_device, tau, V_device, W_device, Height, Width
        )

        # ГРАНИЧНЫЕ УСЛОВИЯ
        lbm_boundary_kernel[blockspergrid, threadsperblock](
            F_tmp_device, F_out_device, Height, Width
        )

        # Меняем местами указатели на массивы (новое состояние -> F_device)
        F_device, F_tmp_device = F_tmp_device, F_device

    # Ждем завершения всех CUDA-ядрер, чтобы корректно измерить время
    cuda.synchronize()
    frame_dt = time.time() - t0
    print(f"GPU frame: {frame_dt:.6f} s for {STEPS_PER_FRAME} steps ({frame_dt/STEPS_PER_FRAME:.6e} s/step)")

    # Копируем распределение на хост только один раз на кадр
    F_host = F_device.copy_to_host()

    # Вычисляем текущие макроскопические поля на CPU
    Rho = Mode0(F_host)
    Ux_loc, Uy_loc = Mode1(F_host)
    Ux_loc /= Rho
    Uy_loc /= Rho

    # Обновляем изображение завихренности
    fluidImage.set_array(curl(Ux_loc, Uy_loc))
    return (fluidImage, barrierImage)

# Создаем и сохраняем анимацию
anim = matplotlib.animation.FuncAnimation(
    fig,
    nextFrame,
    frames=NUM_FRAMES,
    interval=30,  # задержка между кадрами в мс
    blit=True     # blitting ускоряет отрисовку
)

# plt.show()

from IPython.display import HTML
HTML(anim.to_html5_video())


