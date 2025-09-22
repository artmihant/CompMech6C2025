import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

method_name = 'RK4'

# Model Parameters
AU = 1.496e11
DAY = 86400.0
YEAR = 365.25*DAY
G = 6.67430e-11

M_sun = 1.98847e30
M_earth = 5.9722e24
M_mars = 6.4171e23
M_jup = 1.89813e27

names_with_J = ["Sun", "Earth", "Mars", "Jupiter"]
names_without_J = ["Sun", "Earth", "Mars"]
colors = {
    "Sun":   "gold",
    "Earth": "dodgerblue",
    "Mars":  "orangered",
    "Jupiter":"sienna"
}

# Initial conditions
def circ_state(r):
    v = np.sqrt(G*M_sun/r)
    return np.array([r, 0.0], dtype=np.float64), np.array([0.0, v], dtype=np.float64)

a_earth = AU
a_mars = 1.524*AU
a_jup = 5.204*AU

masses_with_J = np.array([M_sun, M_earth, M_mars, M_jup], dtype=np.float64)

r0_with_J_list = [np.array([0.0, 0.0], dtype=np.float64)]
v0_with_J_list = [np.array([0.0, 0.0], dtype=np.float64)]

re, ve = circ_state(a_earth)
rm, vm = circ_state(a_mars)
rj, vj = circ_state(a_jup)

r0_with_J_list += [re, rm, rj]
v0_with_J_list += [ve, vm, vj]

r0_with_J = np.stack(r0_with_J_list)
v0_with_J = np.stack(v0_with_J_list)

mask_without_J = np.array([True, True, True, False])
masses_without_J = masses_with_J[mask_without_J]
r0_without_J = r0_with_J[mask_without_J]
v0_without_J = v0_with_J[mask_without_J]

# Differential equations:
# x' = v
# v' = a(r), a_i = G * sum_{j != i} m_j * (r_j - r_i) / |r_j - r_i|^3
def f(X, m):
    N = X.shape[0] // 2
    r = X[:N]
    v = X[N:]
    a = np.zeros_like(r, dtype=np.float64)
    for i in range(N):
        ri = r[i]
        diff = r - ri
        dist2 = np.sum(diff*diff, axis=1)
        mask = np.arange(N) != i
        diff_m = diff[mask]
        dist = np.sqrt(dist2[mask])
        inv3 = 1.0 / (dist**3)
        a[i] = G * np.sum((m[mask, None] * diff_m) * inv3[:, None], axis=0)
    dX = np.zeros_like(X)
    dX[:N] = v
    dX[N:] = a
    return dX


def simulate_nbody(r0, v0, m, dt, steps, method_name):
    N = r0.shape[0]
    X = np.zeros((2*N, 2), dtype=np.float64)
    X[:N] = r0
    X[N:] = v0

    traj = np.zeros((steps+1, N, 2), dtype=np.float64)
    traj[0] = r0
    for i in range(steps):
        if method_name == 'RK4':
            k1 = f(X, m)
            k2 = f(X + 0.5*dt*k1, m)
            k3 = f(X + 0.5*dt*k2, m)
            k4 = f(X + dt*k3, m)
            X = X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        if method_name == 'Euler':
            X = X + dt * f(X, m)

        traj[i+1] = X[:N]
    return traj

# Time mesh
t_0 = 0.0
t_1_years = 12.0
dt_days   = 1.0

t_1 = t_1_years * YEAR
dt  = dt_days * DAY
N_steps = int(np.round((t_1 - t_0)/dt))

t = np.linspace(t_0, t_1, N_steps+1)

# Calculation of two trajectorie: with and without Jupiter
traj_with_J = simulate_nbody(r0_with_J, v0_with_J, masses_with_J, dt, N_steps, method_name=method_name)
traj_without_J = simulate_nbody(r0_without_J, v0_without_J, masses_without_J, dt, N_steps, method_name=method_name)

# Calculating Mars and Earth semi axes with and without Jupiter
earth_traj_with_J = traj_with_J[:, 1] - traj_with_J[:, 0]
a_earth_with_J = np.max(np.abs(earth_traj_with_J[:, 0]))
b_earth_with_J = np.max(np.abs(earth_traj_with_J[:, 1]))

mars_traj_with_J = traj_with_J[:, 2] - traj_with_J[:, 0]
a_mars_with_J = np.max(np.abs(mars_traj_with_J[:, 0]))
b_mars_with_J = np.max(np.abs(mars_traj_with_J[:, 1]))

earth_traj_without_J = traj_without_J[:, 1] - traj_without_J[:, 0]
a_earth_without_J = np.max(np.abs(earth_traj_without_J[:, 0]))
b_earth_without_J = np.max(np.abs(earth_traj_without_J[:, 1]))

mars_traj_without_J = traj_without_J[:, 2] - traj_without_J[:, 0]
a_mars_without_J = np.max(np.abs(mars_traj_without_J[:, 0]))
b_mars_without_J = np.max(np.abs(mars_traj_without_J[:, 1]))

# Calculating orbit deviations
earth_a_dev = abs(a_earth_with_J - a_earth_without_J) / a_earth_with_J
earth_b_dev = abs(b_earth_with_J - b_earth_without_J) / b_earth_with_J
mars_a_dev = abs(a_mars_with_J - a_mars_without_J) / a_mars_with_J
mars_b_dev = abs(b_mars_with_J - b_mars_without_J) / b_mars_with_J

print(f"Earth semi-major axis deviation: {earth_a_dev*100:.4f}%")
print(f"Earth semi-minor axis deviation: {earth_b_dev*100:.4f}%")
print(f"Mars semi-major axis deviation: {mars_a_dev*100:.4f}%")
print(f"Mars semi-minor axis deviation: {mars_b_dev*100:.4f}%")

def show_orbit_animation(traj, names, title_suffix):
    fig, ax = plt.subplots()
    lines = []
    points = []

    K = len(names)
    cmap = plt.cm.viridis
    palette = cmap(np.linspace(0, 1, K))

    xy_all = traj.reshape(-1, 2)
    max_range = 1.1 * np.max(np.linalg.norm(xy_all, axis=1))
    max_range = max(max_range, 1.2*AU)

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)
    ax.set(xlabel='X, m', ylabel='Y, m', title=f'Orbit Animation {title_suffix}')

    ax.plot(0, 0, 'ko', markersize=6)

    for i, name in enumerate(names):
        if name in colors:
            color = colors[name]
        else:
            color = palette[i]
        line,  = ax.plot([], [], '-', color=color, linewidth=1.5)
        point, = ax.plot([], [], 'o', color=color, markersize=6)
        lines.append(line)
        points.append(point)

    legend_handles = points
    legend_labels  = names
    ax.legend(legend_handles, legend_labels, loc='best')

    trail_len = max(50, min(int(0.05 * (YEAR/dt) * 365), 3000))

    def loop_animation(i):
        artists = []
        start = max(0, i - trail_len)
        seg = slice(start, i+1)
        for k in range(len(names)):
            xy = traj[seg, k]
            lines[k].set_data(xy[:,0], xy[:,1])
            xy_cur = traj[i, k]
            points[k].set_data([xy_cur[0]], [xy_cur[1]])
            artists.extend([lines[k], points[k]])
        return artists

    ani = animation.FuncAnimation(
        fig=fig,
        func=loop_animation,
        frames=N_steps,
        interval=2,
        blit=True,
        repeat=True,
        repeat_delay=500
    )
    plt.show()

show_orbit_animation(traj_with_J, names_with_J, title_suffix="(Sun–Earth–Mars–Jupiter)")
show_orbit_animation(traj_without_J, names_without_J, title_suffix="(Sun–Earth–Mars)")

