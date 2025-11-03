import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 1.0
c = 1.0
Nx = 201
T = 2.5

def triangular_pluck(x, x0=0.5, w=0.5, h=0.5):
    left = x0 - w/2
    right = x0 + w/2
    u = np.zeros_like(x)
    m1 = (x >= left) & (x <= x0)
    m2 = (x > x0) & (x <= right)
    if np.any(m1):
        u[m1] = h * (x[m1] - left) / max(x0 - left, 1e-15)
    if np.any(m2):
        u[m2] = h * (right - x[m2]) / max(right - x0, 1e-15)
    return u

def energy(u_prev, u_now, dx, dt, c):
    ut = (u_now - u_prev) / dt
    ux = np.zeros_like(u_now)
    ux[1:-1] = (u_now[2:] - u_now[:-2]) / (2*dx)
    return 0.5 * np.sum(ut**2 + (c**2) * ux**2) * dx

def simulate(sigma, Nx=Nx, L=L, c=c, T=T, x0=0.5, w=0.5, h=0.5, cap=50.0):
    x = np.linspace(0.0, L, Nx)
    dx = x[1] - x[0]
    dt = sigma * dx / c
    Nt = int(np.floor(T / dt))
    u0 = triangular_pluck(x, x0=x0, w=w, h=h)
    g0 = np.zeros_like(x)
    u_prev = u0.copy()
    u_now = u0.copy()
    u_next = u_now.copy()
    for j in range(1, Nx - 1):
        u_next[j] = u_now[j] + 0.5*(sigma**2)*(u_now[j+1] - 2*u_now[j] + u_now[j-1]) + dt*g0[j]
    u_next[0] = 0.0
    u_next[-1] = 0.0
    u_prev, u_now = u_now, u_next
    times = [dt]
    amps = [np.max(np.abs(u_now))]
    Es = [energy(u_prev, u_now, dx, dt, c)]
    unstable = False
    for n in range(1, Nt):
        u_next = np.empty_like(u_now)
        for j in range(1, Nx - 1):
            u_next[j] = 2*u_now[j] - u_prev[j] + (sigma**2)*(u_now[j+1] - 2*u_now[j] + u_now[j-1])
        u_next[0] = 0.0
        u_next[-1] = 0.0
        u_prev, u_now = u_now, u_next
        t = (n+1)*dt
        times.append(t)
        amps.append(np.max(np.abs(u_now)))
        Es.append(energy(u_prev, u_now, dx, dt, c))
        if not np.isfinite(u_now).all() or np.max(np.abs(u_now)) > cap:
            unstable = True
            break
    return {
        "x": x, "dx": dx, "dt": dt, "Nt": len(times), "u": u_now, "u_prev": u_prev,
        "times": np.array(times), "amps": np.array(amps), "energies": np.array(Es),
        "unstable": unstable, "sigma": sigma
    }

def animate_sigma(sigma, ylim=(-0.7,0.7), cap_stop=200.0):
    x = np.linspace(0.0, L, Nx)
    dx = x[1]-x[0]
    dt = sigma * dx / c
    Nt = int(np.floor(T/dt))
    u_prev = triangular_pluck(x, w=0.5, h=0.5)
    u_now = u_prev.copy()
    u_next = u_now.copy()
    for j in range(1, Nx-1):
        u_next[j] = u_now[j] + 0.5*(sigma**2)*(u_now[j+1]-2*u_now[j]+u_now[j-1])
    u_next[0]=0.0; u_next[-1]=0.0
    u_prev, u_now = u_now, u_next
    fig, ax = plt.subplots(figsize=(8,4.5))
    line, = ax.plot(x, u_now, lw=2)
    ax.set_xlim(0, L)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    title = ax.set_title(f"Анимация, CFL={sigma:.2f}  t={1*dt:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    step = 1
    def update(_):
        nonlocal u_prev, u_now, step
        u_next = np.empty_like(u_now)
        for j in range(1, Nx - 1):
            u_next[j] = 2*u_now[j] - u_prev[j] + (sigma**2)*(u_now[j+1] - 2*u_now[j] + u_now[j-1])
        u_next[0] = 0.0
        u_next[-1] = 0.0
        u_prev, u_now = u_now, u_next
        step += 1
        line.set_ydata(u_now)
        title.set_text(f"Анимация, CFL={sigma:.2f}  t={step*dt:.3f}")
        if not np.isfinite(u_now).all() or np.max(np.abs(u_now)) > cap_stop:
            ani.event_source.stop()
        return line,
    ani = FuncAnimation(fig, update, frames=Nt, interval=20, blit=True)
    return ani

sigma_list = [0.5, 0.9, 1.0]
results = {s: simulate(s) for s in sigma_list}

plt.figure(figsize=(8,4.5))
for s in sigma_list:
    r = results[s]
    lbl = f"CFL={s:.2f}" + (" (unstable)" if r["unstable"] else "")
    plt.plot(r["times"], r["amps"], label=lbl)
plt.xlabel("t")
plt.ylabel("max |u(x,t)|")
plt.title("Эволюция амплитуды при разных CFL")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8,4.5))
for s in sigma_list:
    r = results[s]
    lbl = f"CFL={s:.2f}" + (" (unstable)" if r["unstable"] else "")
    plt.semilogy(r["times"], r["energies"], label=lbl)
plt.xlabel("t")
plt.ylabel("Энергия")
plt.title("Энергия решения при разных CFL")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

ani_stable = animate_sigma(0.9)
unstable_sigmas = [s for s, r in results.items() if r["unstable"]]
anis_unstable = [animate_sigma(sigma, ylim=(-1.5,1.5)) for sigma in unstable_sigmas]

plt.show()
