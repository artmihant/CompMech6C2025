import time
import matplotlib.pyplot as plt, matplotlib.animation
import numpy as np
import numba as nb
from numba import vectorize

""" 
Метод решеточных уравнений Больцмана (Lattice Boltzmann Method, LBM)
с распараллеливанием шага столкновения через numba.vectorize(target="parallel")
"""

""" Зададим константы и свойства задачи """

STEPS_PER_FRAME = 100

Viscosity = 0.01                                # кинематическая вязкость жидкости, м²/с
Height, Width = 80, 200                         # размеры решетки (узлов по y и x)

BarrierCenter = Height//2, Height//2            # центр круглого препятствия
BarrierRadius = Height//10                      # радиус препятствия

U0 = np.array([0.1, 0])                         # начальная скорость потока (в долях скорости звука)

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
    return barrier


""" 
Зададим свойства шаблона решетки D2Q9 
"""

D = 2  # Размерность модели (2D)
Q = 9  # Число дискретных скоростей в шаблоне

# Массив дискретных скоростей: V[q] = (vx, vy)
V = np.array([
    [-1, 1],[ 0, 1],[ 1, 1],   # SW, S, SE
    [-1, 0],[ 0, 0],[ 1, 0],   # W, C, E
    [-1,-1],[ 0,-1],[ 1,-1]    # NW, N, NE
], dtype=np.float64)

# Весовые коэффициенты для равновесного распределения
W = np.array([
    1/36, 1/9, 1/36,
    1/9,  4/9, 1/9,
    1/36, 1/9, 1/36
], dtype=np.float64)

C = 1/3**0.5  # Скорость звука в решеточных единицах: c_s = 1/√3
C2 = C*C

# Вспомогательные массивы для векторизованного столкновения (расширены до размерности решётки)
Vx = V[:, 0].reshape(Q, 1, 1)
Vy = V[:, 1].reshape(Q, 1, 1)
Wb = W.reshape(Q, 1, 1)


def InitBarrier():
    """ 
    Создаем маски для граничных условий на барьере (bounce-back).
    Возвращает массив из 9 масок, как в CPU-версии, но в этой реализации
    мы будем использовать только центральную маску barrier[4].
    """
    barrierC = BarrierShape()

    barrierN = np.roll(barrierC,  1, axis=0)
    barrierS = np.roll(barrierC, -1, axis=0)
    barrierE = np.roll(barrierC,  1, axis=1)
    barrierW = np.roll(barrierC, -1, axis=1)
    barrierNE = np.roll(barrierN,  1, axis=1)
    barrierNW = np.roll(barrierN, -1, axis=1)
    barrierSE = np.roll(barrierS,  1, axis=1)
    barrierSW = np.roll(barrierS, -1, axis=1)

    return np.array([
        barrierNW, barrierN, barrierNE,
        barrierW,  barrierC, barrierE,
        barrierSW, barrierS, barrierSE
    ])


def F_stat(Ux, Uy, Rho):
    """ 
    Равновесное распределение Максвелла-Больцмана (дискретное).
    """
    UV = np.zeros((Q, Height, Width))
    for q in range(Q):
        UV[q] = (V[q,0]*Ux + V[q,1]*Uy)/C2

    U2 = (Ux**2 + Uy**2)/C2

    f_stat = np.zeros((Q, Height, Width))
    for q in range(Q):
        f_stat[q] = Rho * W[q] * (1 + UV[q] + UV[q]**2/2 - U2/2)

    return f_stat


def Mode0(f):
    """ 0-й момент — плотность. """
    mode = np.zeros((Height, Width))
    for q in range(Q):
        mode += f[q]
    return mode


def Mode1(f):
    """ 1-й момент — импульс. """
    mode = np.zeros((D, Height, Width))
    for q in range(Q):
        for d1 in range(D):
            mode[d1] += f[q]*V[q,d1]
    return mode


@vectorize(['float64(float64, float64, float64, float64, float64, float64, float64, float64, float64)'],
           target='parallel', nopython=True)
def collision_update(fi, rho, ux, uy, vx, vy, wi, cs2, tau):
    """
    Векторизованное (Numba) обновление компоненты распределения fi
    по схеме BGK:
        f_new = f + (f_eq - f)/tau
    """
    if rho <= 0.0:
        return fi

    u2 = (ux*ux + uy*uy) / cs2
    uv = (vx*ux + vy*uy) / cs2
    feq = rho * wi * (1.0 + uv + 0.5*uv*uv - 0.5*u2)
    return fi + (feq - fi) / tau


def iter(f, f_out, barrier):
    """
    Один временной шаг алгоритма LBM (CPU),
    где шаг столкновения реализован через numba.vectorize(target="parallel").
    """
    # ============= ЭТАП 1: STREAMING (ПЕРЕНОС) =============
    (fNW, fN, fNE, fW, fC, fE, fSW, fS, fSE) = f

    for y in range(Height-1,0,-1):
        fN[y]  = fN[y-1]
        fNE[y] = fNE[y-1]
        fNW[y] = fNW[y-1]

    fS[:-1]  = fS[1:]
    fSE[:-1] = fSE[1:]
    fSW[:-1] = fSW[1:]

    fE[:,1:]  = fE[:,:-1]
    fNE[:,1:] = fNE[:,:-1]
    fSE[:,1:] = fSE[:,:-1]

    fW[:,:-1]  = fW[:,1:]
    fNW[:,:-1] = fNW[:,1:]
    fSW[:,:-1] = fSW[:,1:]

    # ============= ЭТАП 2: BOUNCE-BACK НА БАРЬЕРЕ ===========
    (bNW, bN, bNE, bW, bC, bE, bSW, bS, bSE) = barrier

    fN[bN]   = fS[bC]
    fS[bS]   = fN[bC]
    fE[bE]   = fW[bC]
    fW[bW]   = fE[bC]
    fNE[bNE] = fSW[bC]
    fNW[bNW] = fSE[bC]
    fSE[bSE] = fNW[bC]
    fSW[bSW] = fNE[bC]

    # ============= ЭТАП 3: ВЫЧИСЛЕНИЕ МАКРОСКОПИКИ =========
    Rho = Mode0(f)
    Ux_loc, Uy_loc = Mode1(f)
    Ux_loc /= Rho
    Uy_loc /= Rho

    # ============= ЭТАП 4: COLLISION (VECTORIZE/PARALLEL) ==
    tau = 0.5 + Viscosity/C2

    # Вызов vectorize-ядра по всей решетке и всем направлениям
    # Все аргументы имеют форму (Q, Height, Width)
    f[:] = collision_update(
        f,
        Rho[np.newaxis, :, :],
        Ux_loc[np.newaxis, :, :],
        Uy_loc[np.newaxis, :, :],
        Vx,
        Vy,
        Wb,
        np.full_like(f, C2),
        np.full_like(f, tau)
    )

    # ============= ЭТАП 5: ГРАНИЧНЫЕ УСЛОВИЯ ===============
    f[:,0,:]  = f_out[:,0,:]
    f[:,-1,:] = f_out[:,-1,:]
    f[:,:,0]  = f_out[:,:,0]
    f[:,:,-1] = f_out[:,:,-1]


def curl(ux, uy):
    """ Завихренность поля скорости. """
    return (np.roll(uy,-1,axis=1) - np.roll(uy,1,axis=1)
            - np.roll(ux,-1,axis=0) + np.roll(ux,1,axis=0))


barrier = InitBarrier()

F = F_stat(Ux, Uy, Rho)
F_out = F_stat(Ux, Uy, Rho)

fig, ax = plt.subplots()

fluidImage = ax.imshow(curl(Ux, Uy), origin='lower',
                        norm=plt.Normalize(-.1,.1),
                        cmap=plt.get_cmap('jet'),
                        interpolation='none')

bImageArray = np.zeros((Height, Width, 4), np.uint8)
bImageArray[barrier[4],3] = 100
barrierImage = plt.imshow(bImageArray, origin='lower', interpolation='none')

def nextFrame(_):
    """
    Один кадр анимации: STEPS_PER_FRAME итераций LBM.
    """
    global F

    t0 = time.time()
    for _ in range(STEPS_PER_FRAME):
        iter(F, F_out, barrier)
    frame_dt = time.time() - t0
    print(f"NB-parallel frame: {frame_dt:.6f} s for {STEPS_PER_FRAME} steps ({frame_dt/STEPS_PER_FRAME:.6e} s/step)")

    Rho_loc = Mode0(F)
    Ux_loc, Uy_loc = Mode1(F)
    Ux_loc /= Rho_loc
    Uy_loc /= Rho_loc

    fluidImage.set_array(curl(Ux_loc, Uy_loc))
    return (fluidImage, barrierImage)

animate = matplotlib.animation.FuncAnimation(fig, nextFrame, interval=20, blit=True)
plt.show()



