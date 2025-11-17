import time
import matplotlib.pyplot as plt
import numba

width, height = 1920*2, 1080*2
max_iter = 128

x_min, x_max = -2.2, 1
y_min, y_max = (x_min-x_max)/2, (x_max-x_min)/2

z0 = complex(0, 0)

@numba.njit
def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, z0):
    divergence_step = []

    for y in range(height):
        row = []
        divergence_step.append(row)
        for x in range(width):
            c = complex(
                xmin + (xmax-xmin)*x/width, 
                ymin + (ymax-ymin)*y/height
            ) 

            z = z0

            iter = 0

            for i in range(max_iter):
                if abs(z) > 2:
                    iter = i
                    break
                z = z * z + c
            else:
                iter = max_iter

            row.append(iter)

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

