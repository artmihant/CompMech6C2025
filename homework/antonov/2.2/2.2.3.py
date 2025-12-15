""" # Задание 2.2: Задача трёх тел """

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as anim
from scipy.optimize import fsolve
from IPython.display import HTML

""" # Теория по задаче """

"""
## Различные начальные параметры
$r_i$ - начальное положение i-го тела

$v_i$ - начальная скорость i-го тела

$m_i$ - масса i-го тела

$G$ - гравитационная постоянная
"""

""" ## Различные конфигурации моделирования """

prm_vosem = {
    'r1' : np.array([0.970, -0.243], dtype=float),
    'r2' : np.array([-0.970, 0.243], dtype=float),
    'r3' : np.array([ 0.0, 0.0], dtype=float),
    'v1' : np.array([0.4662036850, 0.4323657300], dtype=float),
    'v2' : np.array([0.4662036850, 0.4323657300], dtype=float),
    'v3' : np.array([-0.93240737, -0.86473146], dtype=float),
    'm1' : 1,
    'm2' : 1,
    'm3' : 1,
    'G'  : 1  # 6.67430e-11 гравитационная постоянная
}
kolcia ={
    'r1' : np.array([0, 1], dtype=float),
    'r2' : np.array([-np.cos(np.pi/6), -np.sin(np.pi/6)], dtype=float),
    'r3' : np.array([ np.cos(np.pi/6), -np.sin(np.pi/6)], dtype=float),
    'v1' : np.array([-1, 0], dtype=float)*0.5,
    'v2' : np.array([ np.sin(np.pi/6), -np.cos(np.pi/6)], dtype=float)*0.5,
    'v3' : np.array([ np.sin(np.pi/6),  np.cos(np.pi/6)], dtype=float)*0.5,
    'm1' : 1.0,
    'm2' : 1.0,
    'm3' : 1.0,
    'G'  : 1  # 6.67430e-11 гравитационная постоянная
}
telo_uletelo = {
    'r1': np.array([1.5, 0.0], dtype=float),
    'r2': np.array([0.0, -1.0], dtype=float),
    'r3': np.array([-1.5, 0.0], dtype=float),
    'v1': np.array([0.0, 0.4], dtype=float),
    'v2': np.array([0.0, 1], dtype=float),
    'v3': np.array([0.0, -0.4], dtype=float),
    'm1': 2.0,
    'm2': 0.1,
    'm3': 2.0,
    'G' : 1  # 6.67430e-11 гравитационная постоянная
}

""" ## Время моделирования и пармаметры обезразмеривания """

# ЗАДАНИЕ НАЧАЛЬНЫХ ПАРАМЕТРОВ
p = prm_vosem # начальные параметры системы
t0 = 0           # время старта
t1 = 300          # время финиша
#PHYSICS
#обезразмеривающие параметры
tau = 1  # секунды в дне
q = 1    # масса земли (кг)
l = 1    # масштаб расстояния (м)

""" ## Параметры вычислений """

# NUMERICS
dt = 0.05 # шаг по времени

""" ## Параметры анимации """

anim_time = 50 # время на анимацию
fps = 5 # кол-во кадров в ссекунду

def animate(r,fps,anim_time):
    step = 1 + math.floor(len(r)/(fps*anim_time))
    r = r[::step,]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    tr_runge = (
        ax.plot( r[:1,0] , r[:1,1] , color="green", linestyle="--", label='Траектория 1-го тела')[0],
        ax.plot( r[:1,2] , r[:1,3] , color="blue", linestyle="--", label='Траектория 2-го тела')[0],
        ax.plot( r[:1,4] , r[:1,5] , color="red", linestyle="--", label='Траектория 3-го тела')[0],
        ax.scatter(r[0,0],r[0,1], color="black", marker="o",label='Положение 1-го тела'),
        ax.scatter(r[0,2],r[0,3], color="black", marker="*",label='Положение 2-го тела'),
        ax.scatter(r[0,4],r[0,5], color="black", marker="p",label='Положение 3-го тела')
    )
    def update(frame):
        tr_runge[0].set_data(r[:frame+1,0], r[:frame+1,1])
        tr_runge[1].set_data(r[:frame+1,2], r[:frame+1,3])
        tr_runge[2].set_data(r[:frame+1,4], r[:frame+1,5])
        tr_runge[3].set_offsets(r[frame, :2])
        tr_runge[4].set_offsets(r[frame,2:4])
        tr_runge[5].set_offsets(r[frame,4:6])
        ax.relim()
        ax.autoscale_view()
        ax.axis('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Анимированные траектории тел', fontsize=14, fontweight='bold')
        ax.legend()
        return (tr_runge[0] , tr_runge[1] , tr_runge[2] , tr_runge[3], tr_runge[4], tr_runge[5])

    animt = anim.FuncAnimation(
        fig, 
        update, 
        frames=len(r),
        interval=1000/fps,  
        blit=True,
        repeat=True
    )
    
    plt.close(fig)
    return HTML(animt.to_jshtml())

r"""
## Уравнения описывающие систему из трёх тел
$$
\begin{cases}
    \dot{r_1} = v_1\\
    \dot{r_2} = v_2\\
    \dot{r_3} = v_3\\
    \dot{v_1} = G\cdot\frac{m_2}{|r_{12}|^3}\cdot\overrightarrow{r_{12}} + G\cdot\frac{m_3}{|r_{13}|^3}\cdot\overrightarrow{r_{13}}\\
    \dot{v_2} = G\cdot\frac{m_1}{|r_{21}|^3}\cdot\overrightarrow{r_{21}} + G\cdot\frac{m_3}{|r_{23}|^3}\cdot\overrightarrow{r_{23}}\\
    \dot{v_3} = G\cdot\frac{m_1}{|r_{31}|^3}\cdot\overrightarrow{r_{31}} + G\cdot\frac{m_2}{|r_{32}|^3}\cdot\overrightarrow{r_{32}}
\end{cases}
$$
"""

def f(U, m1, m2, m3, G):
    ans = np.zeros(12)
    r12 = U[2:4] - U[ :2]
    r23 = U[4:6] - U[2:4]
    ans[ :2] = U[6:8]
    ans[2:4] = U[8:10]
    ans[4:6] = U[10:12]
    ans[6:8] = G*m2/np.power(np.linalg.norm(r12),3)*r12 + G*m3/np.power(np.linalg.norm(r12+r23),3)*(r12+r23)
    ans[8:10] = -G*m1/np.power(np.linalg.norm(r12),3)*r12 + G*m3/np.power(np.linalg.norm(r23),3)*r23
    ans[10:12] = -G*m2/np.power(np.linalg.norm(r23),3)*r23 - G*m1/np.power(np.linalg.norm(r12+r23),3)*(r12+r23)
    return ans

""" # Численные методы использующиеся в работе """

r"""
## Схема Рунге-Кутты 4-го порядка

$$ y_{n+1} = y_n + \frac{k_1+2k_2+2k_3+k_4}{6}\cdot dt $$

$k_1=f(x_n)$

$k_2=f(x_n+\frac{k_1}{2}\cdot dt)$

$k_3=f(x_n+\frac{k_2}{2}\cdot dt)$

$k_4=f(x_n+k_3\cdot dt)$
"""

def RK4(U, dt, m1, m2, m3, G):
    k1 = f(U, m1, m2, m3, G)
    k2 = f( U + k1*dt/2, m1, m2, m3, G)
    k3 = f( U + k2*dt/2, m1, m2, m3, G)
    k4 = f( U + k3*dt, m1, m2, m3, G)
    return U + (k1 + 2*k2 + 2*k3 + k4)*dt/6

r"""
## Явный метод Адамса

$$ y_{n+1} = y_n + dt \cdot \sum_{j=0}^{k-1} b_j f_{n-j} $$

, где $ k $ порядок метода. В работе используетс метод 4-го порядка для со следующими коэффициэнтами:

$ b_0 = \frac{55}{24}$

$ b_1 = -\frac{59}{24}$

$ b_2 = \frac{37}{24}$

$ b_3 = -\frac{9}{24}$

Для старта расчёта на первых трёх шагах используется метод Рунге-Кутты.
"""

def explicit_adams4(U, f_hist, dt, m1, m2, m3, G, frame):
    if frame < 3:
        y_next = RK4(U, dt, m1, m2, m3, G)
        return y_next, f(y_next, m1, m2, m3, G)
    y_next = U + dt/24*(55*f_hist[frame]-59*f_hist[frame-1]+37*f_hist[frame-2]-9*f_hist[frame-3])
    return y_next, f(y_next, m1, m2, m3, G)

r"""
## Неявный метод Адамса

$$ y_{n+1} = y_n + dt \cdot \sum_{j=0}^{k-1} b_j f_{n+1-j} $$

, где $ k $ порядок метода. Изменения по сравнению с явным методом заключаются в дополнительном слагаемом $ f_{n+1} $. В работе используетс метод 4-го порядка для со следующими коэффициэнтами:

$ b_0 = \frac{9}{24}$

$ b_1 = \frac{19}{24}$

$ b_2 = -\frac{5}{24}$

$ b_3 = \frac{1}{24}$

Для старта расчёта на первых трёх шагах используется метод Рунге-Кутты.
"""

def implicit_adams4(U, f_hist, dt, m1, m2, m3, G, frame):
    if frame < 3:
        y_next = RK4(U, dt, m1, m2, m3, G)
        return y_next, f(y_next, m1, m2, m3, G)
    y_predict = explicit_adams4(U, f_hist, dt, m1, m2, m3, G, frame)[0]
    def implicit_eq(y_next):
            f_next = f(y_next, m1, m2, m3, G)
            return y_next - U - (dt / 24) * (
                9 * f_next +
                19 * f_hist[frame] -
                5 * f_hist[frame-1] +
                1 * f_hist[frame-2]
            )
    # Решаем нелинейное уравнение
    y_next = fsolve(implicit_eq, y_predict)
    return y_next, f(y_next, m1, m2, m3, G)

r"""
## Метод Верлета

$$ \frac{U_{n+1} - U_n}{\Delta t} = V_{x_{n+\frac{1}{2}}} $$

$$ \frac{V_{n+\frac{1}{2}} - V_{n-\frac{1}{2}}}{\Delta t} = a_{x_n} $$

Для задания скорости на первом полушаге используется яформула:

$$ V_{\frac{1}{2}} = V_0 + \frac{dt}{2} \cdot a_0 $$

Ускорения вычисляются согласно формулам 4-6 системы уравнений.
"""

def verlet(U, V_half, dt, m1, m2, m3, G):
    U_next = U + dt*V_half
    f_next = f(U_next, m1, m2, m3, G)
    V_next_half = np.zeros(12)
    V_next_half[0:2] = V_half[0:2] + dt * f_next[6:8]
    V_next_half[2:4] = V_half[2:4] + dt * f_next[8:10]
    V_next_half[4:6] = V_half[4:6] + dt * f_next[10:12]
    return U_next, V_next_half

""" ## Вычисления """

def pars_and_initiat(prm, t0, t1, dt, log=False):
    r1 = prm['r1'] / l
    r2 = prm['r2'] / l
    r3 = prm['r3'] / l
    v1 = prm['v1'] * tau / l
    v2 = prm['v2'] * tau / l
    v3 = prm['v3'] * tau / l
    m1 = prm['m1'] / q
    m2 = prm['m2'] / q
    m3 = prm['m3'] / q
    G  = prm['G'] * q * tau**2 / l**3

    if log:
        print(f"Масштабированные параметры:")
        print(f"r1 = {r1}, r2 = {r2}, r3 = {r3}")
        print(f"v1 = {v1}, v2 = {v2}, v3 = {v3}")
        print(f"m1 = {m1}, m2 = {m2}, m3 = {m3}")
        print(f"G = {G}")
    nsteps = math.ceil((t1 - t0) / dt)  # кол-во шагов
    if log:
        print(f"Количество шагов: {nsteps}")
    
    #PREPROCESSING
    r = np.zeros((nsteps+1,12))

    #INITIAL STATE
    r[0,:2] = r1
    r[0,2:4] = r2
    r[0,4:6] = r3
    r[0,6:8] = v1
    r[0,8:10] = v2
    r[0,10:12] = v3
    return r, nsteps, m1, m2, m3, G

def comp_RK4(prm, t0, t1, dt, log=False):
    r, nsteps, m1, m2, m3, G = pars_and_initiat(prm, t0, t1, dt, log)
    progress_step = nsteps // 10
    for frame in range(nsteps):
        r[frame+1] = RK4(r[frame], dt, m1, m2, m3, G)
        if log and ((frame + 1) % progress_step == 0 or frame == 0):
            progress = (frame + 1) / nsteps * 100
            print(f"Прогресс: {frame + 1}/{nsteps} ({progress:.1f}%)")
    return r, nsteps
def comp_explicit_adams4(prm, t0, t1, dt, log=False):
    r, nsteps, m1, m2, m3, G = pars_and_initiat(prm, t0, t1, dt, log)
    f_hist = np.zeros_like(r)
    f_hist[0,:] = f(r[0], m1, m2, m3, G)
    progress_step = nsteps // 10
    for frame in range(nsteps):
        r[frame+1], f_hist[frame+1,:] = explicit_adams4(r[frame], f_hist, dt, m1, m2, m3, G, frame)
        if log and ((frame + 1) % progress_step == 0 or frame == 0):
            progress = (frame + 1) / nsteps * 100
            print(f"Прогресс: {frame + 1}/{nsteps} ({progress:.1f}%)")
    return r, nsteps
def comp_implicit_adams4(prm, t0, t1, dt, log=False):
    r, nsteps, m1, m2, m3, G = pars_and_initiat(prm, t0, t1, dt, log)
    f_hist = np.zeros_like(r)
    f_hist[0,:] = f(r[0], m1, m2, m3, G)
    progress_step = nsteps // 10
    for frame in range(nsteps):
        r[frame+1], f_hist[frame+1,:] = implicit_adams4(r[frame], f_hist, dt, m1, m2, m3, G, frame)
        if log and ((frame + 1) % progress_step == 0 or frame == 0):
            progress = (frame + 1) / nsteps * 100
            print(f"Прогресс: {frame + 1}/{nsteps} ({progress:.1f}%)")
    return r, nsteps
def comp_verlet(prm, t0, t1, dt, log=False):
    r, nsteps, m1, m2, m3, G = pars_and_initiat(prm, t0, t1, dt, log)
    V_half_hist = np.zeros_like(r)
    f_0 = f(r[0], m1, m2, m3, G)
    V_half_hist[0, :2] = r[0,6:8] + dt/2*f_0[6:8]
    V_half_hist[0,2:4] = r[0,8:10] + dt/2*f_0[8:10]
    V_half_hist[0,4:6] = r[0,10:12] + dt/2*f_0[10:12]
    progress_step = nsteps // 10
    for frame in range(nsteps):
        r[frame+1], V_half_hist[frame+1,:] = verlet(r[frame], V_half_hist[frame], dt, m1, m2, m3, G)
        r[frame+1,6:12] = (V_half_hist[frame+1,:6] + V_half_hist[frame,:6])/2
        if log and ((frame + 1) % progress_step == 0 or frame == 0):
            progress = (frame + 1) / nsteps * 100
            print(f"Прогресс: {frame + 1}/{nsteps} ({progress:.1f}%)")
    return r, nsteps

""" # Зелёный уровень """

r, nsteps = comp_verlet(p,t0,t1,dt,True)

fig, axs = plt.subplots(figsize=(10, 4))
v_1_norm = np.array([ np.linalg.norm(v) for v in r[:,6 :8 ] ])
v_2_norm = np.array([ np.linalg.norm(v) for v in r[:,8 :10] ])
v_3_norm = np.array([ np.linalg.norm(v) for v in r[:,10:12] ])
t_space = np.linspace(t0,t1,nsteps+1)
axs.plot(t_space , v_1_norm , label='Норма скорости 1-го тела' , color = "green")
axs.plot(t_space , v_2_norm , label='Норма скорости 2-го тела' , color = "blue")
axs.plot(t_space , v_3_norm , label='Норма скорости 3-го тела' , color = "red")
axs.legend()
axs.set_title('Зависимость скорости от времени', fontsize=14, fontweight='bold')
axs.set_xlabel('T')
axs.set_ylabel('|V|')
plt.tight_layout()
plt.show()

""" # Жёлтый уровень """

#FIGURE
animate(r,fps,anim_time*10)

fig, axs = plt.subplots(figsize=(10, 4))

axs.plot( r[:,0] , r[:,1] , label='Траектория 1-го тела' , color = "green" ,linestyle="--")
axs.plot( r[:,2] , r[:,3] , label='Траектория 2-го тела' , color = "blue" , linestyle="--")
axs.plot( r[:,4] , r[:,5] , label='Траектория 3-го тела' , color = "red" , linestyle="--")
axs.legend()
axs.set_title('Полные траектории тел', fontsize=14, fontweight='bold')
axs.set_xlabel('X')
axs.set_ylabel('Y')
plt.tight_layout()
plt.show()

""" # Красный уровень """

""" ## Рассмотрим как различные методы ведут себя по сравнению с методом Рунге 4-го порядка """

p = prm_vosem
t0 = 0
t1 = 300
r_RK4, nsteps = comp_RK4(p,t0,t1,dt)
r_verlet, nsteps = comp_verlet(p,t0,t1,dt)
r_e_adams, nsteps = comp_explicit_adams4(p,t0,t1,dt)
r_i_adams, nsteps = comp_implicit_adams4(p,t0,t1,dt)
t_space = np.linspace(t0,t1,nsteps+1)

""" ### Поведение энергии """

def comp_E(U, m1, m2, m3, G):
    r12 = U[:2] - U[2:4]
    r13 = U[:2] - U[4:6]
    r23 = U[2:4] - U[4:6]
    l12 = np.linalg.norm(r12)
    l13 = np.linalg.norm(r13)
    l23 = np.linalg.norm(r23)
    E1 = m1*(U[6]*U[6] + U[7]*U[7])/2 - G*m1*( m2/l12 + m3/l13 )
    E2 = m2*(U[8]*U[8] + U[9]*U[9])/2 - G*m2*( m1/l12 + m3/l23 )
    E3 = m3*(U[10]*U[10] + U[11]*U[11])/2 - G*m3*( m1/l13 + m2/l23 )
    E = E1 + E2 + E3
    return E, E1, E2, E3

E_RK4 = np.array([ np.array(comp_E(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_RK4[:, :12] ])
E_e_adams = np.array([ np.array(comp_E(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_e_adams[:, :12] ])
E_i_adams = np.array([ np.array(comp_E(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_i_adams[:, :12] ])
E_verlet = np.array([ np.array(comp_E(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_verlet[:, :12] ])

""" #### Сравнение Верлета с РК4 """

fig, axs = plt.subplots(figsize=(10, 6))
axs.plot(t_space,E_RK4[:,0]-E_verlet[:,0],color='k')
axs.set_title('Расница в полной энернии между Верлетом и РК4 с течением времени')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.grid()
plt.show(fig)

# fig, axs = plt.subplots(figsize=(10, 6))
# axs.plot(t_space,E_RK4[:,1]-E_verlet[:,1],color='r',label='Энергия 1 тела')
# axs.set_title('Расница в энернии 1-го тела между Верлетом и РК4 с течением времени')
# axs.set_xlabel('t')
# axs.set_ylabel('E')
# axs.grid()
# plt.show(fig)

# fig, axs = plt.subplots(figsize=(10, 6))
# axs.plot(t_space,E_RK4[:,2]-E_verlet[:,2],color='g',label='Энергия 2 тела')
# axs.set_title('Расница в энернии 2-го тела между Верлетом и РК4 с течением времени')
# axs.set_xlabel('t')
# axs.set_ylabel('E')
# axs.grid()
# plt.show(fig)

# fig, axs = plt.subplots(figsize=(10, 6))
# axs.plot(t_space,E_RK4[:,3]-E_verlet[:,3],color='b',label='Энергия 3 тела')
# axs.set_title('Расница в энернии 3-го тела между Верлетом и РК4 с течением времени')
# axs.set_xlabel('t')
# axs.set_ylabel('E')
# axs.grid()

# plt.show(fig)

""" #### Сравнение явного метода Адамса с РК4 """

fig, axs = plt.subplots(figsize=(10, 6))
axs.plot(t_space,E_e_adams[:,0]-E_verlet[:,0],color='k')
axs.set_title('Расница в полной энернии между явным методом Адамса и РК4 с течением времени')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.grid()
plt.show(fig)

# fig, axs = plt.subplots(figsize=(10, 6))
# axs.plot(t_space,E_e_adams[:,1]-E_verlet[:,1],color='r',label='Энергия 1 тела')
# axs.set_title('Расница в энернии 1-го тела между явным методом Адамса и РК4 с течением времени')
# axs.set_xlabel('t')
# axs.set_ylabel('E')
# axs.grid()
# plt.show(fig)

# fig, axs = plt.subplots(figsize=(10, 6))
# axs.plot(t_space,E_e_adams[:,2]-E_verlet[:,2],color='g',label='Энергия 2 тела')
# axs.set_title('Расница в энернии 2-го тела между явным методом Адамса и РК4 с течением времени')
# axs.set_xlabel('t')
# axs.set_ylabel('E')
# axs.grid()
# plt.show(fig)

# fig, axs = plt.subplots(figsize=(10, 6))
# axs.plot(t_space,E_e_adams[:,3]-E_verlet[:,3],color='b',label='Энергия 3 тела')
# axs.set_title('Расница в энернии 3-го тела между явным методом Адамса и РК4 с течением времени')
# axs.set_xlabel('t')
# axs.set_ylabel('E')
# axs.grid()

# plt.show(fig)

""" #### Сравнение неявного метода Адамса с РК4 """

fig, axs = plt.subplots(figsize=(10, 6))
axs.plot(t_space,E_i_adams[:,0]-E_verlet[:,0],color='k')
axs.set_title('Расница в полной энернии между неявным методом Адамса и РК4 с течением времени')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.grid()
plt.show(fig)

# fig, axs = plt.subplots(figsize=(10, 6))
# axs.plot(t_space,E_i_adams[:,1]-E_verlet[:,1],color='r',label='Энергия 1 тела')
# axs.set_title('Расница в энернии 1-го тела между неявным методом Адамса и РК4 с течением времени')
# axs.set_xlabel('t')
# axs.set_ylabel('E')
# axs.grid()
# plt.show(fig)

# fig, axs = plt.subplots(figsize=(10, 6))
# axs.plot(t_space,E_i_adams[:,2]-E_verlet[:,2],color='g',label='Энергия 2 тела')
# axs.set_title('Расница в энернии 2-го тела между неявным методом Адамса и РК4 с течением времени')
# axs.set_xlabel('t')
# axs.set_ylabel('E')
# axs.grid()
# plt.show(fig)

# fig, axs = plt.subplots(figsize=(10, 6))
# axs.plot(t_space,E_i_adams[:,3]-E_verlet[:,3],color='b',label='Энергия 3 тела')
# axs.set_title('Расница в энернии 3-го тела между неявным методом Адамса и РК4 с течением времени')
# axs.set_xlabel('t')
# axs.set_ylabel('E')
# axs.grid()

# plt.show(fig)

"""
#### Выводы

Метод Вертеля быстро расходится с методом РК4. Это можно заметить по тому как растёт разница энергий, образуя конус. Явный метод Адамса также со временем расходится с РК4. Но в данном случае расхождениие имеет логарифмический или параболический рост. Неявный метод Адамса первые 100 единиц времени держится близко с РК4, но после начинает расходится подобно методу Верлета.
"""

""" ### Поведение углового момента """

def comp_L(U, m1, m2, m3, G):
    L1 = m1*(U[0]*U[7] - U[1]*U[6])
    L2 = m2*(U[2]*U[9] - U[3]*U[8])
    L3 = m3*(U[4]*U[11] - U[5]*U[10])
    L = L1 + L2 + L3
    return L, L1, L2, L3

L_RK4 = np.array([ np.array(comp_L(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_RK4[:, :12] ])
L_e_adams = np.array([ np.array(comp_L(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_e_adams[:, :12] ])
L_i_adams = np.array([ np.array(comp_L(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_i_adams[:, :12] ])
L_verlet = np.array([ np.array(comp_L(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_verlet[:, :12] ])

""" #### Сравнение Верлета с РК4 """

fig, axs = plt.subplots(figsize=(10, 6))
axs.plot(t_space,L_RK4[:,0]-L_verlet[:,0],color='k')
axs.set_title('Расница в полном угловом моменте между Верлетом и РК4 с течением времени')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.grid()
plt.show(fig)

""" #### Сравнение явного метода Адамса с РК4 """

fig, axs = plt.subplots(figsize=(10, 6))
axs.plot(t_space,L_RK4[:,0]-L_e_adams[:,0],color='k')
axs.set_title('Расница в полном угловом моменте между явным методом Адамса и РК4 с течением времени')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.grid()
plt.show(fig)

""" #### Сравнение неявного метода Адамса с РК4 """

fig, axs = plt.subplots(figsize=(10, 6))
axs.plot(t_space,L_RK4[:,0]-L_i_adams[:,0],color='k')
axs.set_title('Расница в полном угловом моменте между неявным методом и РК4 с течением времени')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.grid()
plt.show(fig)

"""
#### Вывод

Угловой момент совпадают в случае РК4, неявного Адамса, Верлета, и незначительно отличаются в случае явного метода Адамса. Также можно заметить, что в случае явного метода Адамса наблюдается рост разности моментов, в то время как в Верлете и неявном Адамсе разность ведет себя одинаково периодически в напротяжении всего периода наблюдения.
"""

r"""
### Показатель Ляпунова

Показатель Ляпунова расчитывается следующим образом.

1) Инициализация
   $$ X_0 = X(0) $$
   $$ \tilde{X_0} = X(0) + \delta_0 \cdot e, \quad ||e|| = 1$$
2) Интегрирование
   $$ \dot{X} = f(X) $$
   $$ \dot{\tilde{X}} = f(\tilde{X}) $$
3) Фиксация возмущения на $k$-ом шаге
   $$ \delta X_k = \tilde{X}_k - X_k $$
4) Перенормировка
   $$ \tilde{X}_k = X_k + \delta_0 \frac{\delta X_k}{||\delta X_k||} $$
5) Вычисление параметра Ляпунова
   $$\lambda = \frac{1}{T}\sum_{k=0}^{N} \ln\frac{\delta X_k}{\delta_0}$$
"""

def pok_lyap_RK4(p, t0, t1, dt, delta=1.e-1, log=False): 
    delta0 = np.random.random(12)
    delta0 = delta*delta0 / np.linalg.norm(delta0)
    
    r1, nsteps, m1, m2, m3, G = pars_and_initiat(p, t0, t1, dt, log)
    r2 = np.copy(r1)
    r2[0] += delta0

    deltas = np.zeros(nsteps+1)
    deltas[0] = delta
    
    progress_step = nsteps // 10
    for frame in range(nsteps):
        r1[frame+1] = RK4(r1[frame], dt, m1, m2, m3, G)
        r2[frame+1] = RK4(r2[frame], dt, m1, m2, m3, G)
        deltas[frame+1] = np.linalg.norm(r2[frame+1] - r1[frame+1])
        r2[frame+1] = r1[frame+1] + delta*(r2[frame+1] - r1[frame+1])/deltas[frame+1]
        
        
        if log and ((frame + 1) % progress_step == 0 or frame == 0):
            progress = (frame + 1) / nsteps * 100
            print(f"Прогресс: {frame + 1}/{nsteps} ({progress:.1f}%)")
    lyap = 0
    for i in range(len(deltas)):
        lyap += np.log(deltas[i]/delta)
    lyap /= t1-t0
    return lyap, r1, r2, deltas, nsteps

p = prm_vosem
lyap, r1, r2, deltas, nsteps = pok_lyap_RK4(p,t0,t1,dt, 1.e-1)
print(f'Показатель Ляпунова: {lyap}')

fig, axs = plt.subplots(figsize=(10, 4))

axs.plot( r1[:,0] , r1[:,1] , label='Траектория 1-го тела' , color = "green" ,linestyle="--")
axs.plot( r2[:,0] , r2[:,1] , label='Траектория 1-го тела с возмущением' , color = "blue" , linestyle="--")
axs.legend()
axs.set_title('Устойчивость движения 1-го тела', fontsize=14, fontweight='bold')
axs.set_xlabel('X')
axs.set_ylabel('Y')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(figsize=(10, 4))

axs.plot( r1[:,2] , r1[:,3] , label='Траектория 2-го тела' , color = "green" ,linestyle="--")
axs.plot( r2[:,2] , r2[:,3] , label='Траектория 2-го тела с возмущением' , color = "blue" , linestyle="--")
axs.legend()
axs.set_title('Устойчивость движения 2-го тела', fontsize=14, fontweight='bold')
axs.set_xlabel('X')
axs.set_ylabel('Y')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(figsize=(10, 4))

axs.plot( r1[:,4] , r1[:,5] , label='Траектория 3-го тела' , color = "green" ,linestyle="--")
axs.plot( r2[:,4] , r2[:,5] , label='Траектория 3-го тела с возмущением' , color = "blue" , linestyle="--")
axs.legend()
axs.set_title('Устойчивость движения 3-го тела', fontsize=14, fontweight='bold')
axs.set_xlabel('X')
axs.set_ylabel('Y')
plt.tight_layout()
plt.show()

"""
#### Вывод

Для движения по восьмерке показатель Ляпунова равен 0.32. Это говорит о сильной не устойчивости системы, что видно по графикам. При не большом отклонении( разность в первном занке после запятой) траектории наглядно расходятся.
"""

"""
## Заключение

В результате работы мы сравнили 4 численных метода для расчета задачи трёх. Оказалось, что:
- Расность энергии системы всех методов с РК4 растёт с течением времени, но у каждого метода рост имеет свои особенности
- Угловой момент же не показывает сильных различий в выборе метода
- При помощи показателя Ляпунова получилось показать хаотичность системы
"""
