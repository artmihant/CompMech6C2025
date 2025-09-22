import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as anim
import helper as h
import analit as an
import Init_param as prm

#PHYSICS
#обезразмеривающие параметры
tau = 3600 * 24  # секунды в дне
q = 5.97e24      # масса земли (кг)
l = 1e8         # масштаб расстояния (м)

# задание параметров системы
p = prm.prm_Earth_Sun
r1 = p['r1']
r2 = p['r2']
v1 = p['v1']
v2 = p['v2']
m1 = p['m1']
m2 = p['m2']


G = 6.67430e-11  # гравитационная постоянная
t0 = 0
t1 = 0.5          # время симуляции в днях
# Масштабирование переменных
r1, r2, v1, v2, m1, m2, G = h.unmer(r1, r2, v1, v2, m1, m2, G, l, tau, q)
# Характеристики центра масс
v_cm = (m1*v1 + m2*v2)/(m1+m2)
r_cm = (m1*r1 + m2*r2)/(m1+m2)
# базис в плоскости проходяей через центр масс и
# нормалью, совпадающей по направлению с направлением движения центра масс,
# так как скорость центра масс постоянна
e1 , e2 , e3 = h.orthonormal_basis_from_normal(v_cm)

print(f"Масштабированные параметры:")
print(f"r1 = {r1}, r2 = {r2}")
print(f"v1 = {v1}, v2 = {v2}")
print(f"m1 = {m1}, m2 = {m2}")
print(f"G = {G}")

# NUMERICS
dt = 0.001  # шаг по времени
nsteps = math.ceil((t1 - t0) / dt) #кол-во шагов
print(f"Количество шагов: {nsteps}")

#PREPROCESSING
r_an  = an.two_body_analytic_trajectory(r1, v1, r2, v2, m1, m2, G,e1,e2,e3,
                                       t0=t0, t1=t1, n_pts=nsteps+1, plot=False)
r_num_euler = np.zeros((nsteps+1,12))
r_num_runge = np.zeros((nsteps+1,12))

r_pr_euler = np.zeros((nsteps+1,4))
r_pr_runge = np.zeros((nsteps+1,4))

r_pr_euler[0] = [ r1@e1 , r1@e2 , r2@e1 , r2@e2 ]
r_pr_runge[0] = [ r1@e1 , r1@e2 , r2@e1 , r2@e2 ]

h.start(r_num_euler,r1,r2,v1,v2)
h.start(r_num_runge,r1,r2,v1,v2)

#FIGURE
fig, ax = plt.subplots()
ax.axis('equal')

tr_euler = h.inic_lines(ax,r_pr_euler) # траектория для схемы Эйлера
tr_runge = h.inic_lines(ax,r_pr_runge) # траектория для схемы Рунге-Кута 4 порядка
tr_an = (
    # ax.plot(r_an['proj']['x1'],r_an['proj']['y1'],color="black", linestyle="--")[0],
    # ax.plot(r_an['proj']['x2'],r_an['proj']['y2'],color="purple", linestyle=":")[0],
    ax.scatter(r_an['proj']['x1'],r_an['proj']['y1'],color="black", marker=".",label='Траектория 1-ro тела Аналитика'),
    ax.scatter(r_an['proj']['x2'],r_an['proj']['y2'],color="purple", marker=".",label='Траектория 2-ro тела Аналитика'),
)
tr_runge[0].set_color('red')
tr_runge[2].set_color("red")
tr_runge[1].set_color("blue")
tr_runge[3].set_color("blue")
tr_runge[4].set_color("purple")
tr_runge[5].set_color("purple")
tr_runge[0].set_label('Траектория 1-ro тела Рунге')
tr_runge[1].set_label('Траектория 2-ro тела Рунге')
tr_runge[4].set_label('Положение 1-ro тела Рунге')
tr_runge[5].set_label('Положение 2-ro тела Рунге')

tr_euler[0].set_label('Траектория 1-ro тела Эйлер')
tr_euler[1].set_label('Траектория 2-ro тела Эйлер')
tr_euler[4].set_label('Положение 1-ro тела Эйлер')
tr_euler[5].set_label('Положение 2-ro тела Эйлер')

def f(U):
    ans = np.zeros(12)
    r = U[3:6] - U[ :3]
    ans[ :3] = U[6:9]
    ans[3:6] = U[9:12]
    ans[6:9] = G*m2/np.power(np.linalg.norm(r),3)*r
    ans[9:12] = -m1/m2*ans[6:9]
    return ans
def RK4( U ):
    k1 = f(U)
    k2 = f( U + k1*dt/2 )
    k3 = f( U + k2*dt/2 )
    k4 = f( U + k3*dt )
    return k1 + 2*k2 + 2*k3 + k4
def init_anim():
    def init_anim_help(tr,r):
        tr[0].set_data   (r[:1,0],r[:1,1])
        tr[1].set_data   (r[:1,2],r[:1,3])
        tr[2].set_offsets(r[:1, :2])
        tr[3].set_offsets(r[:1,2:4])
        tr[4].set_offsets(r[0,:2])
        tr[5].set_offsets(r[0,2:4])
    init_anim_help(tr_euler,r_pr_euler)
    init_anim_help(tr_runge,r_pr_runge)
    return (tr_euler[0] , tr_euler[1] , tr_euler[2] , tr_euler[3], tr_euler[4], tr_euler[5],
            tr_runge[0] , tr_runge[1] , tr_runge[2] , tr_runge[3], tr_runge[4], tr_runge[5],
            tr_an[0], tr_an[1])
def update(frame):
    r_num_euler[frame+1] = r_num_euler[frame] + f(r_num_euler[frame])*dt
    r_num_runge[frame+1] = r_num_runge[frame] + RK4( r_num_runge[frame] )*dt/6
    def update_anim_help(tr,r):
        tr[0].set_data(r[:frame+2,0], r[:frame+2,1])
        tr[1].set_data(r[:frame+2,2], r[:frame+2,3])
        tr[2].set_offsets(r[:frame+2, :2])
        tr[3].set_offsets(r[:frame+2, 2:4])
        tr[4].set_offsets(r[frame+1,:2])
        tr[5].set_offsets(r[frame+1,2:4])
    r_pr_euler[frame+1] = np.array([ r_num_euler[frame+1,:3]@e1 , r_num_euler[frame+1,:3]@e2 , r_num_euler[frame+1,3:6]@e1 , r_num_euler[frame+1,3:6]@e2])
    r_pr_runge[frame+1] = np.array([ r_num_runge[frame+1,:3]@e1 , r_num_runge[frame+1,:3]@e2 , r_num_runge[frame+1,3:6]@e1 , r_num_runge[frame+1,3:6]@e2])
    update_anim_help(tr_euler,r_pr_euler)
    update_anim_help(tr_runge,r_pr_runge)
    ax.relim()
    ax.autoscale_view()
    # print( f"{r_num[frame+1,:4]}, {np.linalg.norm(r_num[frame+1,4:6]-r_num[frame,4:6])}, {np.linalg.norm(r_num[frame+1,6:8]-r_num[frame,6:8])}" )
    return (tr_euler[0] , tr_euler[1] , tr_euler[2] , tr_euler[3], tr_euler[4], tr_euler[5],
            tr_runge[0] , tr_runge[1] , tr_runge[2] , tr_runge[3], tr_runge[4], tr_runge[5],
            tr_an[0], tr_an[1])

ani = anim.FuncAnimation(   fig=fig,
                            func=update,
                            init_func=init_anim,
                            frames=nsteps,
                            interval=1,
                            repeat=False )
plt.legend()
plt.show()

def compute_energy_and_L(r_num, m1, m2, G):
    n = r_num.shape[0]
    E = np.zeros(n)
    L = np.zeros(n)
    for i in range(n):
        r1 = r_num[i, :3]
        r2 = r_num[i, 3:6]
        v1 = r_num[i, 6:9]
        v2 = r_num[i, 9:12]

        r12 = np.linalg.norm(r1 - r2)

        T = 0.5 * m1 * np.dot(v1, v1) + 0.5 * m2 * np.dot(v2, v2)
        U = - G * m1 * m2 / r12
        E[i] = T + U

        L_vec = m1 * np.cross(r1, v1) + m2 * np.cross(r2, v2)
        L[i] = np.linalg.norm(L_vec)
    return E, L

# Вычисляем энергии и моменты
E_euler, L_euler = compute_energy_and_L(r_num_euler, m1, m2, G)
E_runge, L_runge = compute_energy_and_L(r_num_runge, m1, m2, G)

t_grid = np.linspace(t0, t1, nsteps+1)

fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Энергия
axes[0].plot(t_grid, E_euler, label="Эйлер", color="orange")
axes[0].plot(t_grid, E_runge, label="RK4", color="blue")
axes[0].set_ylabel("Энергия")
axes[0].set_title("Полная энергия")
axes[0].legend()
axes[0].grid()

# Угловой момент
axes[1].plot(t_grid, L_euler, label="Эйлер", color="orange")
axes[1].plot(t_grid, L_runge, label="RK4", color="blue")
axes[1].set_ylabel("|L|")
axes[1].set_title("Угловой момент")
axes[1].legend()
axes[1].grid()

plt.show()

