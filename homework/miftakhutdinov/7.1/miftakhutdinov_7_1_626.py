import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 1.0
c = 1.0

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "figure.figsize": (7, 4),
    "axes.grid": True,
    "grid.alpha": 0.3
})

def initial_pluck(x, A=1.0):
    y = np.zeros_like(x)
    m = x <= 0.5*L
    y[m] = (2*A/L)*x[m]
    y[~m] = (2*A/L)*(L - x[~m])
    return y

def simulate_wave(N, CFL, T, A=1.0, blowup_threshold=50.0):
    dx = L/(N-1)
    dt = CFL*dx/c
    Nt = int(np.floor(T/dt))
    x = np.linspace(0, L, N)
    u0 = initial_pluck(x, A=A)
    g = np.zeros_like(x)
    u1 = u0.copy()
    u1[1:-1] = u0[1:-1] + dt*g[1:-1] + 0.5*(CFL**2)*(u0[2:] - 2*u0[1:-1] + u0[:-2])
    u0[0] = u0[-1] = 0.0
    u1[0] = u1[-1] = 0.0
    U = [u0.copy(), u1.copy()]
    times = [0.0, dt]
    E = []
    for n in range(1, Nt):
        u_nm1 = U[-2]
        u_n = U[-1]
        u_np1 = np.zeros_like(u_n)
        u_np1[1:-1] = 2*u_n[1:-1] - u_nm1[1:-1] + (CFL**2)*(u_n[2:] - 2*u_n[1:-1] + u_n[:-2])
        u_np1[0] = 0.0
        u_np1[-1] = 0.0
        vel = (u_n - u_nm1)/dt
        du = np.diff(u_n)/dx
        E.append(0.5*np.sum(vel**2)*dx + 0.5*(c**2)*np.sum(du**2)*dx)
        U.append(u_np1)
        times.append(times[-1] + dt)
        if np.isnan(u_np1).any() or np.max(np.abs(u_np1)) > blowup_threshold:
            break
    return x, np.array(U), np.array(times), np.array(E), dt, dx

def make_animation(x, U, times, CFL, A=1.0, fps=30, max_frames=240, title_prefix="1D Wave, fixed ends"):
    stride = max(1, int(np.ceil(len(times)/max_frames)))
    idx = np.arange(0, len(times), stride)
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t)$")
    ax.set_title(f"{title_prefix}. CFL = {CFL:.2f}")
    ax.set_xlim(0, L)
    ax.set_ylim(-1.2*A, 1.2*A)
    line, = ax.plot(x, U[0], lw=2)
    txt = ax.text(0.02, 0.92, "", transform=ax.transAxes)
    def update(frame):
        k = idx[frame]
        line.set_ydata(U[k])
        txt.set_text(f"t = {times[k]:.3f}")
        return line, txt
    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000/fps, blit=True)
    # try:
        # from IPython.display import HTML, display
        # display(HTML(anim.to_jshtml()))
        # plt.close(fig)
    # except Exception:
    plt.show()

def plot_energy(times, E, CFL):
    plt.figure()
    plt.plot(times[1:len(E)+1], E, lw=2, label=f"CFL = {CFL:.2f}")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$E(t)$")
    plt.title(f"Discrete energy, CFL = {CFL:.2f}")
    plt.tight_layout()
    plt.show()

def main():
    N = 201
    T = 4.0
    A = 1.0
    CFLS = [0.20, 0.50, 0.90, 1.00, 1.05]
    energy_sets = []
    time_sets = []
    for CFL in CFLS:
        x, U, times, E, dt, dx = simulate_wave(N=N, CFL=CFL, T=T, A=A)
        make_animation(x, U, times, CFL, A=A)
        plot_energy(times, E, CFL)
        energy_sets.append(E)
        time_sets.append(times)
    plt.figure()
    for CFL, times, E in zip(CFLS, time_sets, energy_sets):
        plt.plot(times[1:len(E)+1], E, lw=2, label=f"CFL = {CFL:.2f}")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$E(t)$")
    plt.title("Discrete energy comparison across CFL")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
