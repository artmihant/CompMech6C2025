import time
import numpy as np
import matplotlib.pyplot as plt

width, height = 1920*2, 1080*2
max_iter = 128

x_min, x_max = -2.2, 1
y_min, y_max = (x_min-x_max)/2, (x_max-x_min)/2

z0 = complex(0, 0)


def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, z0):
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    real, imag = np.meshgrid(real, imag)
    c = real + 1j * imag

    z = np.full(c.shape, z0, dtype=np.complex128)

    divergence_step = np.full(c.shape, max_iter, dtype=int)

    mask = divergence_step == max_iter

    for i in range(max_iter):
        
        z[mask] = z[mask] ** 2 + c[mask]

        divergence_step[mask & (np.abs(z) > 2)] = i
        mask = mask & (np.abs(z) < 2)
        pass

    return divergence_step

now = time.time()

image = mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, z0)

print(time.time() - now)

plt.imshow(image, cmap='hot', extent=(x_min, x_max, y_min, y_max))
plt.colorbar(label="Log2(Iters before devergence)")
plt.title("Mandelbrot fractal")
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()

