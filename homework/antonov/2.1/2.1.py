import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as anim

#Различные начальные состояния
prm_1 = {
    'w1' : np.pi/2 ,
    'w2' : 0 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 1 ,
    'm1' : 1,
    'm2' : 1,
    't0' : 0,
    't1' : 10,
    'g' : 9.81
}
prm_2 = {
    'w1' : np.pi/2 ,
    'w2' : np.pi/2 - 0*np.pi/2 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 0.1 ,
    'm1' : 5,
    'm2' : 0.5,
    't0' : 0,
    't1' : 10,
    'g' : 9.81
}
prm_3 = {
    'w1' : 4*np.pi/10 ,
    'w2' : 0 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 1 ,
    'm1' : 1,
    'm2' : 1,
    't0' : 0,
    't1' : 20,
    'g' : 9.81
}
prm_haos_1 = {
    'w2' : 1*np.pi/10 ,
    'w1' : 0 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 1 ,
    'm1' : 1,
    'm2' : 1,
    't0' : 0,
    't1' : 20,
    'g' : 9.81
}
prm_haos_2 = {
    'w2' : 2*np.pi/10 ,
    'w1' : 0 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 1 ,
    'm1' : 1,
    'm2' : 1,
    't0' : 0,
    't1' : 20,
    'g' : 9.81
}
prm_haos_3 = {
    'w2' : 3*np.pi/10 ,
    'w1' : 0 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 1 ,
    'm1' : 1,
    'm2' : 1,
    't0' : 0,
    't1' : 20,
    'g' : 9.81
}
prm_haos_4 = {
    'w2' : 4*np.pi/10 ,
    'w1' : 0 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 1 ,
    'm1' : 1,
    'm2' : 1,
    't0' : 0,
    't1' : 20,
    'g' : 9.81
}
prm_haos_5 = {
    'w2' : 5*np.pi/10 ,
    'w1' : 0 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 1 ,
    'm1' : 1,
    'm2' : 1,
    't0' : 0,
    't1' : 20,
    'g' : 9.81
}
prm_haos_6 = {
    'w2' : 6*np.pi/10 ,
    'w1' : 0 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 1 ,
    'm1' : 1,
    'm2' : 1,
    't0' : 0,
    't1' : 20,
    'g' : 9.81
}
prm_haos_7 = {
    'w2' : 7*np.pi/10 ,
    'w1' : 0 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 1 ,
    'm1' : 1,
    'm2' : 1,
    't0' : 0,
    't1' : 20,
    'g' : 9.81
}
prm_haos_8 = {
    'w2' : 8*np.pi/10 ,
    'w1' : 0 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 1 ,
    'm1' : 1,
    'm2' : 1,
    't0' : 0,
    't1' : 20,
    'g' : 9.81
}
prm_haos_9 = {
    'w2' : 9*np.pi/10 ,
    'w1' : 0 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 1 ,
    'm1' : 1,
    'm2' : 1,
    't0' : 0,
    't1' : 20,
    'g' : 9.81
}
prm_haos_10 = {
    'w2' : 9.99*np.pi/10 ,
    'w1' : 0 ,
    'teta1' : 0,
    'teta2' : 0 ,
    'l1' : 1 ,
    'l2' : 1 ,
    'm1' : 1,
    'm2' : 1,
    't0' : 0,
    't1' : 20,
    'g' : 9.81
}



#PHYSICS
#обезразмеривающие параметры
tau = 1  # секунды в дне
q = 1    # масса земли (кг)
l = 1         # масштаб расстояния (м)
x0 , y0 = ( 0 , 1 ) # положение точки подвеса
time_anim = 10
# NUMERICS
dt = 0.001                          # шаг по времени
interval = 30 #время между кадрами в милисек
t0 = 0
t1 = 20

def func(prm,ax):
    # ЗАДАНИЕ НАЧАЛЬНЫХ ПАРАМЕТРОВ
    p = prm # начальные параметры системы
    # Масштабирование переменных
    w1 = p['w1']
    w2 = p['w2']
    teta1 = p['teta1']
    teta2 = p['teta2']
    l1 = p['l1'] / l
    l2 = p['l2'] / l
    m1 = p['m1'] / q
    m2 = p['m2'] / q
    g = p['g']*tau**2/l
    t0 = p['t0']
    t1 = p['t1']
    nsteps = math.ceil((t1 - t0) / dt)  #кол-во шагов
    print(f"Количество шагов: {nsteps}")

    print(f"Масштабированные параметры:")
    print(f"w1 = {w1}, w2 = {w2}")
    print(f"teta1 = {teta1}, teta2 = {teta2}")
    print(f"l1 = {l1}, l2 = {l2}")
    print(f"m1 = {m1}, m2 = {m2}")
    print(f"g = {g}")

    #PREPROCESSING
    w = np.zeros((nsteps+1,4))
    rr = np.zeros((nsteps+1,4))

    #INITIAL STATE
    w[0,0] = w1
    w[0,1] = w2
    w[0,2] = teta1
    w[0,3] = teta2
    rr[0,:2] = np.array([ l1*np.cos(w[0,0] - np.pi ) , l1*np.sin(w[0,0] - np.pi) ])
    rr[0,2:4] = np.array([ l2*np.cos(w[0,1] - np.pi ) , l2*np.sin(w[0,1] - np.pi) ]) + rr[0,:2]

    #CALCULATING........
    def f(U):
        ans = np.zeros(4)
        ans[0] = U[2]
        ans[1] = U[3]
        ans[2] = (
                -g * (2*m1 + m2) * np.sin(U[0])
                - m2 * g * np.sin(U[0] - 2*U[1])
                - 2 * np.sin(U[0] - U[1]) * m2 * (U[3]**2 * l2 + U[2]**2 * l1 * np.cos(U[0] - U[1]))
            ) / (l1 * (2*m1 + m2 - m2 * np.cos(U[0] - U[1])**2))
        ans[3] = (
                2 * np.sin(U[0] - U[1]) * (
                    U[2]**2 * l1 * (m1 + m2)
                    + g * (m1 + m2) * np.cos(U[0])
                    + U[3]**2 * l2 * m2 * np.cos(U[0] - U[1])
                )
            ) / (l2 * (2*m1 + m2 - m2 * np.cos(U[0] - U[1])**2))
        return ans
    def RK4( U ):
        k1 = f(U)
        k2 = f( U + k1*dt/2 )
        k3 = f( U + k2*dt/2 )
        k4 = f( U + k3*dt )
        return k1 + 2*k2 + 2*k3 + k4

    progress_step = nsteps // 10
    for frame in range(nsteps):
        w[frame+1] = w[frame] + RK4(w[frame]) * dt / 6
        rr[frame+1,:2] = np.array([ l1*np.cos(w[frame+1,0] - np.pi/2 ) , l1*np.sin(w[frame+1,0] - np.pi/2) ])
        rr[frame+1,2:4] = np.array([ l2*np.cos(w[frame+1,1] - np.pi/2 ) , l2*np.sin(w[frame+1,1] - np.pi/2) ]) + rr[frame+1,:2]
        if (frame + 1) % progress_step == 0 or frame == 0:
            progress = (frame + 1) / nsteps * 100
            print(f"Прогресс: {frame + 1}/{nsteps} ({progress:.1f}%)")


    #FIGURE
    skips = math.floor(nsteps/(time_anim*1000/interval)) # сколько кадров надо пропустить чтобы уложиться во временные рамки
    r = rr[::skips] #удаление лишьних кадров
    frames = math.floor(r.size/r[0].size) # кол-во кадров

    ax = axs[0]
    ax.axis([x0-(l1+l2), x0+(l1+l2), y0-(l1+l2), y0+(l1+l2)])
    tr_runge = (
        ax.plot( np.array([x0 , x0 +r[0,0]]) , np.array([y0 , y0 +r[0,1]]) , color="green", linestyle="-")[0],
        ax.plot( np.array([x0 +r[0,0] , x0 +r[0,2]]) , np.array([y0 +r[0,1] , y0 +r[0,3]]) , color="blue", linestyle="-")[0],
        ax.scatter(x0,y0, color="red", marker="*"),
        ax.scatter(x0+r[0,0],y0+r[0,1], color="black", marker="o"),
        ax.scatter(x0+r[0,2],y0+r[0,3], color="black", marker="o")
    )
    return (r , tr_runge , frames, nsteps , w)

fig, axs = plt.subplots(1,2)

r1 , tr_runge1 , frames1, nsteps1, w1 = func(prm_haos_1,axs)
r2 , tr_runge2 , frames2, nsteps2, w2 = func(prm_haos_2,axs)
r3 , tr_runge3 , frames3, nsteps2, w3 = func(prm_haos_3,axs)
r4 , tr_runge4 , frames4, nsteps2, w4 = func(prm_haos_4,axs)
r5 , tr_runge5 , frames5, nsteps2, w5 = func(prm_haos_5,axs)
r6 , tr_runge6 , frames6, nsteps2, w6 = func(prm_haos_6,axs)
r7 , tr_runge7 , frames7, nsteps2, w7 = func(prm_haos_7,axs)
r8 , tr_runge8 , frames8, nsteps2, w8 = func(prm_haos_8,axs)
r9 , tr_runge9 , frames9, nsteps2, w9 = func(prm_haos_9,axs)
r10 , tr_runge10 , frames10, nsteps10, w10 = func(prm_haos_10,axs)

frames = frames1
nsteps = nsteps1
w = w1

pr = [(r1 , tr_runge1),
      (r2 , tr_runge2),
      (r3 , tr_runge3),
      (r4 , tr_runge4),
      (r5 , tr_runge5),
      (r6 , tr_runge6),
      (r7 , tr_runge7),
      (r8 , tr_runge8),
      (r9 , tr_runge9),
      (r10 , tr_runge10)]

def init_anim(r,tr_runge):
    tr_runge[0].set_data   (np.array([x0 , x0 +r[0,0]]) , np.array([y0 , y0 +r[0,1]]))
    tr_runge[1].set_data   (np.array([x0 +r[0,0] , x0 +r[0,2]]) , np.array([y0 +r[0,1] , y0 +r[0,3]]))
    tr_runge[2].set_offsets(np.array([x0,y0]))
    tr_runge[3].set_offsets(np.array([x0+r[0,0],y0+r[0,1]]))
    tr_runge[4].set_offsets(np.array([x0+r[0,2],y0+r[0,3]]))
    return (tr_runge[0] , tr_runge[1] , tr_runge[2] , tr_runge[3], tr_runge[4])

def update(frame,r,tr_runge):
    tr_runge[0].set_data   (np.array([x0 , x0 +r[frame+1,0]]) , np.array([y0 , y0 +r[frame+1,1]]))
    tr_runge[1].set_data   (np.array([x0 +r[frame+1,0] , x0 +r[frame+1,2]]) , np.array([y0 +r[frame+1,1] , y0 +r[frame+1,3]]))
    tr_runge[2].set_offsets(np.array([x0,y0]))
    tr_runge[3].set_offsets(np.array([x0+r[frame+1,0],y0+r[frame+1,1]]))
    tr_runge[4].set_offsets(np.array([x0+r[frame+1,2],y0+r[frame+1,3]]))
    return (tr_runge[0] , tr_runge[1] , tr_runge[2] , tr_runge[3], tr_runge[4])

def init_all():
    al = ()
    for el in pr:
        al += init_anim(el[0],el[1])
    return al


def update_all(frame):
    al = ()
    for el in pr:
        al += update(frame,el[0],el[1])
    return al

ani = anim.FuncAnimation(   fig=fig,
                            func=update_all,
                            init_func=init_all,
                            frames=frames-1,
                            interval=interval,
                            repeat=False )

t_space = np.linspace(t0,t1,nsteps+1)
axs[1].plot( t_space , w[:,0] , color="red", linestyle="-", label="Угол 1-го ребра" )
axs[1].plot( t_space , w[:,1] , color="blue", linestyle="-", label="Угол 2-го ребра" )
axs[1].set_xlabel('Время, с')
axs[1].set_ylabel('Угол, рад') 
axs[1].set_title('Зависимость углов от времени')
axs[1].legend() 

plt.tight_layout()
plt.show()