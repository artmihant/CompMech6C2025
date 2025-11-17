import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 1.0     
c = 1.0     
Nx = 201     
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)

def f(x):  
    return np.where(x <= 0.5, 2 * x, 2 * (1 - x))

def g(x):
    return np.zeros_like(x)

def wave_1d(cfl=0.9, T=2.0):
    r = cfl                 
    dt = r * dx / c
    Nt = int(T / dt) + 1

    u_nm1 = np.zeros(Nx)   
    u_n = f(x)            
    u_np1 = np.zeros(Nx)  

    u_n[0] = u_n[-1] = 0.0

    u_np1[1:-1] = u_n[1:-1] + 0.5 * r**2 * (u_n[2:] - 2*u_n[1:-1] + u_n[:-2])
    u_np1[0] = u_np1[-1] = 0.0

    history = [u_n.copy()]
    times = [0.0]

    for n in range(1, Nt - 1):
        lap = u_np1[2:] - 2*u_np1[1:-1] + u_np1[:-2]
        u_new = 2*u_np1 - u_n
        u_new[1:-1] += r**2 * lap
        u_new[0] = u_new[-1] = 0.0

        if n % max(1, Nt // 150) == 0: 
            history.append(u_np1.copy())
            times.append(n * dt)

        u_n, u_np1 = u_np1, u_new

    return np.array(history), np.array(times)

def animate_wave(cfl=0.9, T=2.0):
    hist, times = wave_1d(cfl, T)
    fig, ax = plt.subplots(figsize=(6, 3))
    line, = ax.plot(x, hist[0])
    ax.set_xlim(0, L)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(f"Волновое уравнение, CFL = {cfl}")

    def update(i):
        line.set_ydata(hist[i])
        ax.set_title(f"t = {times[i]:.2f} (CFL = {cfl})")
        return line,

    ani = FuncAnimation(fig, update, frames=len(hist), interval=40, blit=True)
    plt.show()


animate_wave(cfl=0.9, T=2.0)
