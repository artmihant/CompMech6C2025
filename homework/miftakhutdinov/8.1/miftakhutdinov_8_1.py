import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def solve_laplace_gauss_seidel(nx=121, ny=121, V0=1.0, tol=1e-5, max_iters=20000, verbose=True):
    dx = dy = 1.0
    phi = np.zeros((ny, nx), dtype=float)

    # ГУ
    phi[-1, :] = +V0   
    phi[0,  :] = -V0   
    phi[:, 0]  = 0.0   
    phi[:, -1] = 0.0   

    inv_denom = 1.0 / (2.0/dx**2 + 2.0/dy**2)

    it = 0
    residual = np.inf
    while it < max_iters:
        max_delta = 0.0
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                old = phi[j, i]
                num = (phi[j, i+1] + phi[j, i-1]) / dx**2 + (phi[j+1, i] + phi[j-1, i]) / dy**2
                phi[j, i] = num * inv_denom
                d = abs(phi[j, i] - old)
                if d > max_delta:
                    max_delta = d

        phi[-1, :] = +V0
        phi[0,  :] = -V0
        phi[:, 0]  = 0.0
        phi[:, -1] = 0.0

        residual = max_delta
        it += 1
        
        if residual < tol:
            if verbose:
                print(f"Сошлось за {it} итераций, maxΔ = {residual:.3e}")
            break

    return phi, it, residual


def compute_field(phi, dx=1.0, dy=1.0):
    dphidy, dphidx = np.gradient(phi, dy, dx)
    Ex = -dphidx
    Ey = -dphidy
    return Ex, Ey


def normalize_field(Ex, Ey, eps=1e-12):
    mag = np.sqrt(Ex**2 + Ey**2)
    mag = np.maximum(mag, eps)
    return Ex/mag, Ey/mag


def plot_potential(phi):
    ny, nx = phi.shape
    x = np.arange(nx)
    y = np.arange(ny)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(phi, origin='lower', extent=[x.min(), x.max()-1, y.min(), y.max()-1],
                    aspect='equal', cmap='RdBu_r')
    plt.colorbar(im, label='Потенциал φ')
    plt.title('Карта потенциала φ')
    plt.xlabel('x (узел)')
    plt.ylabel('y (узел)')
    plt.tight_layout()
    plt.show()


def plot_field_quiver(Ex, Ey, normalize=True, step=3):
    ny, nx = Ex.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    U, V = (normalize_field(Ex, Ey) if normalize else (Ex, Ey))

    plt.figure(figsize=(7, 6))
    skip = (slice(None, None, step), slice(None, None, step))
    plt.quiver(X[skip], Y[skip], U[skip], V[skip], pivot='mid',
               scale_units='xy', angles='xy', scale=1)
    plt.title('Векторное поле E = -∇φ (quiver)')
    plt.xlabel('x (узел)')
    plt.ylabel('y (узел)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def plot_field_stream(Ex, Ey):
    ny, nx = Ex.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(7, 6))
    plt.streamplot(X, Y, Ex, Ey, density=1.2, linewidth=1.0, arrowsize=1.2)
    plt.title('Линии поля E (streamplot)')
    plt.xlabel('x (узел)')
    plt.ylabel('y (узел)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def main():
    nx, ny = 121, 121
    V0 = 1.0
    tol = 1e-5
    max_iters = 20000
    verbose = True

    phi, iters, residual = solve_laplace_gauss_seidel(nx, ny, V0, tol, max_iters, verbose)
    Ex, Ey = compute_field(phi)

    plot_potential(phi)                 
    plot_field_quiver(Ex, Ey, True, 3) 

  
if __name__ == '__main__':
    main()
