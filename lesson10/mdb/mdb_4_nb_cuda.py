import time
import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import cuda

width, height = 1920*2, 1080*2

x_min, x_max = -2.2, 1
y_min, y_max = (x_min-x_max)/2, (x_max-x_min)/2


@cuda.jit('void(float64[:], float64[:], int32[:])')
def calc_iter(real, imag, divergence_step):
    k = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if k > divergence_step.size:
        return

    z = complex(0,0)
    c = real[k] + 1j * imag[k]

    for i in range(128):
        if abs(z) > 2:
            divergence_step[k] = i
            return
        z = z * z + c
    
    divergence_step[k] = 128


def mandelbrot(xmin, xmax, ymin, ymax, width, height):
    divergence_step = np.zeros((height,width), dtype='int32')

    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    real, imag = np.meshgrid(real, imag)

    d_real = cuda.to_device(real.ravel())
    d_imag = cuda.to_device(imag.ravel())
    d_divergence_step = cuda.to_device(divergence_step.ravel())
    
    device = cuda.get_current_device()

    tpb = device.WARP_SIZE
    bpg = int(np.ceil((height*width)/tpb)) 

    print(tpb, bpg)

    calc_iter[bpg, tpb](d_real, d_imag, d_divergence_step)

    divergence_step = d_divergence_step.copy_to_host()

    return divergence_step.reshape((height,width)) 

now = time.time()
image = mandelbrot(x_min, x_max, y_min, y_max, width, height)
print(time.time() - now)

plt.imshow(image, cmap='hot', extent=(x_min, x_max, y_min, y_max))
plt.colorbar(label="Iters before devergence")
plt.title("Mandelbrot fractal")
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()

