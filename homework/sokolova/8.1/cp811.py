import numpy as np
import matplotlib.pyplot as plt

nx, ny = 161, 121
V0 = 1.0 
tol = 1e-4 
max_iter = 8000
omega = 1.9 
hx = hy = 1.0
report_every = 200

phi = np.zeros((ny, nx), dtype=float)

plate_width = 40
plate_start = (nx - plate_width) // 2
plate_end = plate_start + plate_width

phi[0, :] = 0.0
phi[-1, :] = 0.0
phi[:, 0] = 0.0
phi[:, -1] = 0.0

lower_plate_y = ny // 3
upper_plate_y = 2 * ny // 3

phi[lower_plate_y, plate_start:plate_end] = -V0
phi[upper_plate_y, plate_start:plate_end] = +V0

fixed = np.zeros_like(phi, dtype=bool)
fixed[0, :] = True; fixed[-1, :] = True
fixed[:, 0] = True; fixed[:, -1] = True
fixed[lower_plate_y, plate_start:plate_end] = True
fixed[upper_plate_y, plate_start:plate_end] = True

def sor(phi, fixed, hx, hy, tol, max_iter, omega, report_every=500):
    ny, nx = phi.shape
    hx2, hy2 = hx*hx, hy*hy
    denom = 2.0*(hx2 + hy2)
    for it in range(1, max_iter+1):
        max_delta = 0.0
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                if fixed[j, i]:
                    continue
                gs = ((phi[j, i+1] + phi[j, i-1]) * hy2 +
                      (phi[j+1, i] + phi[j-1, i]) * hx2) / denom
                new_val = (1 - omega) * phi[j, i] + omega * gs
                delta = abs(new_val - phi[j, i])
                if delta > max_delta:
                    max_delta = delta
                phi[j, i] = new_val
        if max_delta < tol:
            return it, max_delta
    return max_iter, max_delta

iters, final_delta = sor(phi, fixed, hx, hy, tol, max_iter, omega, report_every)

dphidy, dphidx = np.gradient(phi, hy, hx)
Ey = -dphidy
Ex = -dphidx

E_magnitude = np.sqrt(Ex**2 + Ey**2)

plt.figure(figsize=(10, 7))
plt.title("Распределение потенциала φ в плоском конденсаторе")
im = plt.imshow(phi, origin='lower', aspect='auto', cmap='RdBu_r', extent=[0, nx, 0, ny])
plt.colorbar(im, label="Потенциал φ, В")
plt.xlabel("Координата x")
plt.ylabel("Координата y")

plt.axhline(y=lower_plate_y, xmin=plate_start/nx, xmax=plate_end/nx, 
           color='black', linewidth=2, label='Пластины конденсатора')
plt.axhline(y=upper_plate_y, xmin=plate_start/nx, xmax=plate_end/nx, 
           color='black', linewidth=2)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
plt.title("Напряженность электрического поля E = -∇φ")

Ex_norm = np.where(E_magnitude > 0, Ex/E_magnitude, 0)
Ey_norm = np.where(E_magnitude > 0, Ey/E_magnitude, 0)

step = max(nx//30, 1)
Y, X = np.mgrid[0:ny:1, 0:nx:1]

plt.imshow(E_magnitude, origin='lower', aspect='auto', cmap='plasma', 
           alpha=0.7, extent=[0, nx, 0, ny])
plt.colorbar(label="|E|, В/ед.")

plt.quiver(X[::step, ::step], Y[::step, ::step],
           Ex_norm[::step, ::step], Ey_norm[::step, ::step],
           scale=30, color='white', width=0.003)

plt.xlabel("Координата x")
plt.ylabel("Координата y")

plt.axhline(y=lower_plate_y, xmin=plate_start/nx, xmax=plate_end/nx, 
           color='red', linewidth=2, linestyle='--', alpha=0.7)
plt.axhline(y=upper_plate_y, xmin=plate_start/nx, xmax=plate_end/nx, 
           color='red', linewidth=2, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
