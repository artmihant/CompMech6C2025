import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time


def Julia(z, c, max_iter=100):
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter


def Mandelbrot(c, max_iter=100):
    return Julia(0, c, max_iter)


def CreateMandelbrotSet(width=800, height=800, x_min=-2, x_max=1, 
                          y_min=-1.5, y_max=1.5, max_iter=100):
    """
    Создает множество Мандельброта
    """
    # Создаем сетку комплексных чисел
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j*Y
    
    mandelbrot_set = np.zeros((height, width))
    
    print("Вычисление множества Мандельброта...")
    
    for i in range(height):
        if i % 100 == 0:
            print(f"Прогресс: {i}/{height}")
        for j in range(width):
            mandelbrot_set[i, j] = Mandelbrot(C[i, j], max_iter)
    
    return mandelbrot_set


def CreateJuliaSet(c, width=800, height=800, x_min=-2, x_max=2, 
                    y_min=-2, y_max=2, max_iter=100):
    """
    Создает множество Джулиа для заданного параметра c
    """
    # Создаем сетку комплексных чисел
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    julia_set = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            julia_set[i, j] = Julia(Z[i, j], c, max_iter)
    
    return julia_set


def CreateCustomColormap():
    colors = ['#000033', '#000055', '#0000BB', '#0E4C92', 
              '#2E8BC0', '#19D3F3', '#FED766', '#FE4A49', 
              '#FFFFFF']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('Mandelbrot', colors, N=n_bins)
    return cmap


def PlotMandelbrot():
    mandelbrot_set = CreateMandelbrotSet(width=1000, height=1000, max_iter=100)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(mandelbrot_set, extent=[-2, 1, -1.5, 1.5], 
               cmap=CreateCustomColormap(), origin='lower', 
               interpolation='bilinear')
    plt.colorbar(label='Количество итераций до расхождения')
    plt.title('Множество Мандельброта\n$z_{n+1} = z_n^2 + c$', fontsize=16)
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def PlotJuliaSets():
    # Параметры для множеств Джулиа
    Julia_params = [
        {'c': -0.7 + 0.27j, 'name': 'Связное множество\nc = -0.7 + 0.27i'},
        {'c': -0.4 + 0.6j, 'name': 'Дендритная структура\nc = -0.4 + 0.6i'},
        {'c': 0.285 + 0.01j, 'name': 'Квази-связное\nc = 0.285 + 0.01i'},
        {'c': -0.8 + 0.156j, 'name': 'Драконоподобное\nc = -0.8 + 0.156i'},
        {'c': 0.3 + 0.5j, 'name': 'Несвязное множество\nc = 0.3 + 0.5i'},
    ]
    
    fig = plt.figure(figsize=(18, 12))
    
    for idx, params in enumerate(Julia_params, 1):
        print(f"Вычисление множества Джулиа для c = {params['c']}")
        Julia_set = CreateJuliaSet(params['c'], width=600, height=600, max_iter=100)
        
        ax = plt.subplot(2, 3, idx)
        im = ax.imshow(Julia_set, extent=[-2, 2, -2, 2], 
                      cmap=CreateCustomColormap(), origin='lower',
                      interpolation='bilinear')
        ax.set_title(params['name'], fontsize=12)
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Множества Джулиа для различных параметров c', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":    
    print("\n1. Построение множества Мандельброта...")
    PlotMandelbrot()

    print("\n2. Построение множеств Джулиа...")
    PlotJuliaSets()
