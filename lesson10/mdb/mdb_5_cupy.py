import cupy as cp
import matplotlib.pyplot as plt
import time

width, height = 1920*2, 1080*2

x_min, x_max = -2.2, 1
y_min, y_max = (x_min-x_max)/2, (x_max-x_min)/2

def mandelbrot_cupy(xmin, xmax, ymin, ymax, width, height, max_iter=128):
    real = cp.linspace(xmin, xmax, width)
    imag = cp.linspace(ymin, ymax, height)
    real, imag = cp.meshgrid(real, imag)

    c = real + 1j * imag
    z = cp.zeros_like(c)
    steps = cp.zeros(c.shape, dtype=cp.int32)

    for i in range(max_iter):
        mask = (steps == 0) & (cp.abs(z) > 2)
        steps[mask] = i
        z = z*z + c

    steps[steps == 0] = max_iter
    return steps

now = time.time()
img = mandelbrot_cupy(x_min, x_max, y_min, y_max, width, height)
print("GPU time:", time.time() - now)

plt.imshow(cp.asnumpy(img), cmap='hot', extent=(x_min, x_max, y_min, y_max))
plt.colorbar()
plt.show()