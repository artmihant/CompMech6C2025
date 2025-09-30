import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as anim

#Различные начальные состояния
prm_vosem = {# тут нужно G = 1
    'r1' : np.array([0.970, -0.243], dtype=float),
    'r2' : np.array([-0.970, 0.243], dtype=float),
    'r3' : np.array([ 0.0, 0.0], dtype=float),
    'v1' : np.array([0.4662036850, 0.4323657300], dtype=float),
    'v2' : np.array([0.4662036850, 0.4323657300], dtype=float),
    'v3' : np.array([-0.93240737, -0.86473146], dtype=float),
    'm1' : 1,
    'm2' : 1,
    'm3' : 1
}
prm_Earth_Sun_Earth = {
    'r1' : np.array([-1.5e9, 0], dtype=float),
    'r2' : np.array([ 0, 0], dtype=float),
    'r3' : np.array([ 1.5e9, 0], dtype=float),
    'v1' : np.array([0, -5e4], dtype=float),
    'v2' : np.array([0, 1e5], dtype=float),
    'v3' : np.array([0,  5e4], dtype=float),
    'm1' : 5.97e24,
    'm2' : 5.97e29,
    'm3' : 5.97e24
}
prm_Earth_Earth_Earth = {
    'r1' : np.array([-1.5e9, 0], dtype=float),
    'r2' : np.array([ 0, 0], dtype=float),
    'r3' : np.array([ 1.5e9, 0], dtype=float),
    'v1' : np.array([0, -5e4], dtype=float),
    'v2' : np.array([-4e4, 3e4], dtype=float),
    'v3' : np.array([0,  5e4], dtype=float),
    'm1' : 5.97e24,
    'm2' : 5.97e24,
    'm3' : 5.97e24
}
prm_Sun_Sun_Sun = {
    'r1' : np.array([-1.5e9, 0], dtype=float),
    'r2' : np.array([ 0, 0], dtype=float),
    'r3' : np.array([ 1.5e9, 0], dtype=float),
    'v1' : np.array([0, -5e4], dtype=float),
    'v2' : np.array([-3e4, 4e4], dtype=float),
    'v3' : np.array([0,  5e4], dtype=float),
    'm1' : 5.97e29,
    'm2' : 5.97e29,
    'm3' : 5.97e29
}
prm_Earth_Sun_Moon = {
    'r1' : np.array([-1.49e9, 0], dtype=float),
    'r2' : np.array([ 0, 0], dtype=float),
    'r3' : np.array([ -1.4935e9, 0], dtype=float),
    'v1' : np.array([0, -2.9e4], dtype=float),
    'v2' : np.array([0, 0], dtype=float),
    'v3' : np.array([0, -3e4], dtype=float),
    'm1' : 5.97e24,
    'm2' : 2e30,
    'm3' : 5.97e22
}
prm_Earth_Sun_Moon = {
    'r1' : np.array([-149.e9, 0], dtype=float),
    'r2' : np.array([ 0, 0], dtype=float),
    'r3' : np.array([ -149.35e9, 0], dtype=float),
    'v1' : np.array([0, -2.9e4], dtype=float),
    'v2' : np.array([0, 0], dtype=float),
    'v3' : np.array([0, -3e4], dtype=float),
    'm1' : 5.97e24,
    'm2' : 2e30,
    'm3' : 5.97e22
}

# ЗАДАНИЕ НАЧАЛЬНЫХ ПАРАМЕТРОВ
p = prm_vosem # начальные параметры системы
G = 1  # 6.67430e-11 гравитационная постоянная
t0 = 0              
t1 = 5


#PHYSICS
#обезразмеривающие параметры
tau = 1  # секунды в дне
q = 1    # масса земли (кг)
l = 1         # масштаб расстояния (м)

# Масштабирование переменных
r1 = p['r1'] / l
r2 = p['r2'] / l
r3 = p['r3'] / l
v1 = p['v1'] * tau / l
v2 = p['v2'] * tau / l
v3 = p['v3'] * tau / l
m1 = p['m1'] / q
m2 = p['m2'] / q
m3 = p['m3'] / q
G = G * q * tau**2 / l**3

print(f"Масштабированные параметры:")
print(f"r1 = {r1}, r2 = {r2}, r3 = {r3}")
print(f"v1 = {v1}, v2 = {v2}, v3 = {v3}")
print(f"m1 = {m1}, m2 = {m2}, m3 = {m3}")
print(f"G = {G}")

# NUMERICS
dt = 0.1                          # шаг по времени
nsteps = math.ceil((t1 - t0) / dt)  #кол-во шагов
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

#FIGURE

fig, axs = plt.subplots(2,2)

ax = axs[0,0]
ax.axis('equal')
tr_runge = (
    ax.plot( r[:1,0] , r[:1,1] , color="green", linestyle="--", label='Траектория 1-го тела')[0],
    ax.plot( r[:1,2] , r[:1,3] , color="blue", linestyle="--", label='Траектория 2-го тела')[0],
    ax.plot( r[:1,4] , r[:1,5] , color="red", linestyle="--", label='Траектория 3-го тела')[0],
    ax.scatter(r[0,0],r[0,1], color="black", marker="o",label='Положение 1-го тела'),
    ax.scatter(r[0,2],r[0,3], color="black", marker="*",label='Положение 2-го тела'),
    ax.scatter(r[0,4],r[0,5], color="black", marker="p",label='Положение 3-го тела')
)

def f(U):
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
def RK4( U ):
    k1 = f(U)
    k2 = f( U + k1*dt/2 )
    k3 = f( U + k2*dt/2 )
    k4 = f( U + k3*dt )
    return k1 + 2*k2 + 2*k3 + k4

progress_step = nsteps // 10
for frame in range(nsteps):
    r[frame+1] = r[frame] + RK4(r[frame]) * dt / 6
    if (frame + 1) % progress_step == 0 or frame == 0:
        progress = (frame + 1) / nsteps * 100
        print(f"Прогресс: {frame + 1}/{nsteps} ({progress:.1f}%)")

def init_anim():
    tr_runge[0].set_data   (r[:1,0],r[:1,1])
    tr_runge[1].set_data   (r[:1,2],r[:1,3])
    tr_runge[2].set_data   (r[:1,4],r[:1,5])
    tr_runge[3].set_offsets(r[:1, :2])
    tr_runge[4].set_offsets(r[:1,2:4])
    tr_runge[5].set_offsets(r[:1,4:6])
    return (tr_runge[0] , tr_runge[1] , tr_runge[2] , tr_runge[3], tr_runge[4], tr_runge[5])

def update(frame):
    tr_runge[0].set_data(r[:frame+2,0], r[:frame+2,1])
    tr_runge[1].set_data(r[:frame+2,2], r[:frame+2,3])
    tr_runge[2].set_data(r[:frame+2,4], r[:frame+2,5])
    tr_runge[3].set_offsets(r[frame+1, :2])
    tr_runge[4].set_offsets(r[frame+1,2:4])
    tr_runge[5].set_offsets(r[frame+1,4:6])
    ax.relim()
    ax.autoscale_view()
    return (tr_runge[0] , tr_runge[1] , tr_runge[2] , tr_runge[3], tr_runge[4], tr_runge[5])

ani = anim.FuncAnimation(   fig=fig,
                            func=update,
                            init_func=init_anim,
                            frames=nsteps,
                            interval=10,
                            repeat=False )

v_1_norm = np.array([ np.linalg.norm(v) for v in r[:,6 :8 ] ])
v_2_norm = np.array([ np.linalg.norm(v) for v in r[:,8 :10] ])
v_3_norm = np.array([ np.linalg.norm(v) for v in r[:,10:12] ])
t_space = np.linspace(t0,t1,nsteps+1)
axs[1,0].plot(t_space , v_1_norm , label='Норма скорости 1-го тела' , color = "green")
axs[1,0].plot(t_space , v_2_norm , label='Норма скорости 2-го тела' , color = "blue")
axs[1,0].plot(t_space , v_3_norm , label='Норма скорости 3-го тела' , color = "red")
axs[1,0].legend()

axs[0,1].plot( r[:,0] , r[:,1] , label='Траектория 1-го тела' , color = "green" ,linestyle="--")
axs[0,1].plot( r[:,2] , r[:,3] , label='Траектория 2-го тела' , color = "blue" , linestyle="--")
axs[0,1].plot( r[:,4] , r[:,5] , label='Траектория 3-го тела' , color = "red" , linestyle="--")
axs[0,1].legend()
axs[0,0].legend()

axs[0, 0].set_title('Анимированные траектории тел', fontsize=14, fontweight='bold')
axs[0, 1].set_title('Полные траектории тел', fontsize=14, fontweight='bold')
axs[1, 0].set_title('Зависимость скорости от времени', fontsize=14, fontweight='bold')

axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')

axs[1, 0].set_xlabel('T')
axs[1, 0].set_ylabel('|V|')

plt.tight_layout()
plt.show()