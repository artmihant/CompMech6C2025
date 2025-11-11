import numpy as np
import matplotlib.pyplot as plt

nx, ny = 121, 81
V0 = 1.0 
tol = 1e-4 
max_iter = 8000
omega = 1.9 
hx = hy = 1.0
report_every = 200

phi = np.zeros((ny, nx), dtype=float)

phi[0, :]  = -V0
phi[-1, :] = +V0

phi[:, 0]  = 0.0
phi[:, -1] = 0.0

fixed = np.zeros_like(phi, dtype=bool)
fixed[0, :] = True; fixed[-1, :] = True
fixed[:, 0] = True; fixed[:, -1] = True

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

dphidy, dphidx = np.gradient(phi, hy, hx, edge_order=2)
Ey = -dphidy
Ex = -dphidx

plt.figure(figsize=(7, 4.5))
plt.title("Распределение потенциала φ")
im = plt.imshow(phi, origin='lower', aspect='auto')
plt.colorbar(im, label="Потенциал φ")
plt.xlabel("Координата x")
plt.ylabel("Координата y")
plt.tight_layout()
plt.show()

step = max(nx//25, 1)
Y, X = np.mgrid[0:ny:1, 0:nx:1]
plt.figure(figsize=(7, 4.5))
plt.title("Напряженность электрического поля E = -∇φ")
plt.quiver(X[::step, ::step], Y[::step, ::step],
           Ex[::step, ::step], Ey[::step, ::step], scale=50)
plt.xlabel("Координата x")
plt.ylabel("Координата y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4.5))
plt.title("Эквипотенциальные линии φ")
im2 = plt.imshow(phi, origin='lower', aspect='auto')
plt.contour(phi, levels=11, linewidths=0.7)
plt.colorbar(im2, label="Потенциал φ")
plt.xlabel("Координата x")
plt.ylabel("Координата y")
plt.tight_layout()
plt.show()
