import time
import matplotlib.pyplot as plt
import numba
import numpy as np

width, height = 1920*2, 1080*2
max_iter = 128

x_min, x_max = -2.2, 1
y_min, y_max = (x_min-x_max)/2, (x_max-x_min)/2

z0 = complex(0, 0)


@numba.vectorize(['int32(float64, float64, complex128, int32)'], target="parallel")
def calc_iter(real, imag, z0, max_iter):
    c = real + 1j * imag
    z = z0

    for i in range(max_iter):
        if abs(z) > 2:
            return i
        z = z * z + c
    return max_iter 


def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, z0):
    divergence_step = np.zeros((height,width), dtype='int32')

    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    real, imag = np.meshgrid(real, imag)
    # c = real + 1j * imag
    
    divergence_step = calc_iter(real, imag, z0, max_iter)

    return divergence_step 


now = time.time()
image = mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, z0)
print(time.time() - now)

# plt.imshow(image, cmap='hot', extent=(x_min, x_max, y_min, y_max))
# plt.colorbar(label="Iters before devergence")
# plt.title("Mandelbrot fractal")
# plt.xlabel("Re")
# plt.ylabel("Im")
# plt.show()

