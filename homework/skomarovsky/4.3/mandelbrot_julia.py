import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Params
n_iter = 100
N = 600
M = 800


def mandelbrot_iteration(c):
    z = 0.0
    for n in range(n_iter):
        if abs(z) > 2.0:
            return n
        z = z*z + c
    return n_iter

def julia_iteration(z, c):
    for n in range(n_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return n_iter

def plot_mandelbrot():
    # Complex plane mesh
    Re0 = -2.0
    Re1 = 1.0
    Im0 = -1.5
    Im1 = 1.5
    Re = np.linspace(Re0, Re1, N)
    Im = np.linspace(Im0, Im1, M)

    Re, Im = np.meshgrid(Re, Im)
    C = Re + 1j * Im

    mandelbrot_set = np.zeros(C.shape, dtype=int)
    for i in range(N):
        for j in range(M):
            mandelbrot_set[j, i] = mandelbrot_iteration(C[j, i])
    
    colors = [(0, 0, 0), (0, 0, 0.5), (0, 0.5, 1), (0.5, 1, 1), 
              (1, 1, 0.5), (1, 0.5, 0), (0.5, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('mandelbrot', colors, n_iter)
    
    plt.figure(figsize=(12, 9))
    plt.imshow(mandelbrot_set, extent=[Re0, Re1, Im0, Im1], cmap=cmap, origin='lower')
    plt.colorbar(label='Number of Iters')
    plt.title('Mandelbrot Set')
    plt.xlabel('Re')
    plt.ylabel('Im')
    
    return mandelbrot_set, C

def plot_julia_sets(c_values):
    
    # Complex plane mesh
    Re0 = -2.0
    Re1 = 2.0
    Im0 = -2.0
    Im1 = 2.0
    Re = np.linspace(Re0, Re1, N)
    Im = np.linspace(Im0, Im1, M)
    Re, Im = np.meshgrid(Re, Im)
    Z = Re + 1j * Im
    
    # Plot for each c
    colors = [(0, 0, 0), (0.2, 0.1, 0.5), (0.4, 0.2, 0.8), (0.6, 0.4, 1), (0.8, 0.8, 1), (1, 1, 1)]
    cmap = LinearSegmentedColormap.from_list('julia', colors, n_iter)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, c in enumerate(c_values):
        julia_set = np.zeros(Z.shape, dtype=int)
        for i in range(N):
            for j in range(M):
                julia_set[j, i] = julia_iteration(Z[j, i], c)
        
        ax = axes[idx]
        im = ax.imshow(julia_set, extent=[Re0, Re1, Im0, Im1], cmap=cmap, origin='lower')
        ax.set_title(f'Julia set for c = {c.real:.3f} + {c.imag:.3f}i')
        ax.set_xlabel('Re')
        ax.set_ylabel('Im')
    
    for idx in range(len(c_values), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()

print("Calculating Mandelbrot set")
mandelbrot_set, C = plot_mandelbrot()
    
print("Calculating Julia set")
c_values = [ -0.7 + 0.27j, -0.4 + 0.6j, 0.285 + 0.01j, -0.8 + 0.156j, -0.745 + 0.113j, 0.3 + 0.5j ]
plot_julia_sets(c_values)

plt.show()
    