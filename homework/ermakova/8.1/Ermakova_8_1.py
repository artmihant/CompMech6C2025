
import numpy as np
import matplotlib.pyplot as plt

# геометрия и сетка 
Lx, Ly = 1.0, 1.0   # размеры области 
Nx, Ny = 81, 81     # числа узлов по x и y 
V = 1.0             # потенциал пластин по модулю

dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

phi = np.zeros((Ny, Nx), dtype=float)

# гр усл
phi[-1, :] = +V
phi[0, :]  = -V
phi[:, 0]  = 0.0
phi[:, -1] = 0.0

tol = 1e-6          # критерий остановы
max_iters = 20000
changes = []        # история max |gradφ| для графика сходимости

inv_dxsq = 1.0/(dx*dx)
inv_dysq = 1.0/(dy*dy)
denom = 2.0*(inv_dxsq + inv_dysq)

for it in range(1, max_iters+1):
    max_change = 0.0
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            old = phi[j, i]
            phi[j, i] = ((phi[j, i+1] + phi[j, i-1]) * inv_dxsq +
                         (phi[j+1, i] + phi[j-1, i]) * inv_dysq) / denom
            diff = abs(phi[j, i] - old)
            if diff > max_change:
                max_change = diff

    phi[-1, :] = +V
    phi[0, :]  = -V
    phi[:, 0]  = 0.0
    phi[:, -1] = 0.0

    changes.append(max_change)
    if max_change < tol:
        break

print(f'Итераций: {it}, финальный max|gradφ| = {changes[-1]:.2e}')


# E = -gradφ 

Ex = np.zeros_like(phi)
Ey = np.zeros_like(phi)

Ex[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2])/(2*dx)
Ey[1:-1, :] = -(phi[2:, :] - phi[:-2, :])/(2*dy)

Ex[:, 0]  = -(phi[:, 1] - phi[:, 0]) / dx
Ex[:, -1] = -(phi[:, -1] - phi[:, -2]) / dx
Ey[0, :]  = -(phi[1, :] - phi[0, :]) / dy
Ey[-1, :] = -(phi[-1, :] - phi[-2, :]) / dy

# графики
X, Y = np.meshgrid(x, y, indexing='xy')

# ф — тепловая карта
plt.figure(figsize=(6,5))
im = plt.imshow(phi, extent=[0, Lx, 0, Ly], origin='lower', aspect='equal')
plt.colorbar(im, label='φ')
plt.title('Распределение потенциала φ (уравнение Лапласа)')
plt.xlabel('x'); plt.ylabel('y')
plt.tight_layout()

# E — векторное поле 
plt.figure(figsize=(6,5))
skip = max(1, Nx//32)
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip],
           Ex[::skip, ::skip], Ey[::skip, ::skip], scale=40)
plt.title('Линии напряжённости E = -gradφ')
plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal')
plt.tight_layout()

# сходимость
plt.figure(figsize=(6,4))
plt.semilogy(np.arange(1, len(changes)+1), changes)
plt.xlabel('Итерация'); plt.ylabel('max |gradφ|')
plt.title('Сходимость метода Гаусса–Зейделя')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()
