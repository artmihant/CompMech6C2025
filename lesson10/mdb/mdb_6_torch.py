import torch
import matplotlib.pyplot as plt
import time

device = "cuda"

width, height = 1920*2, 1080*2

x_min, x_max = -2.2, 1
y_min, y_max = (x_min-x_max)/2, (x_max-x_min)/2

def mandelbrot_torch(xmin, xmax, ymin, ymax, width, height, max_iter=128):
    # Создаём сетку
    real = torch.linspace(xmin, xmax, width, device=device)
    imag = torch.linspace(ymin, ymax, height, device=device)
    real, imag = torch.meshgrid(real, imag, indexing='xy')

    c = real + 1j * imag
    z = torch.zeros_like(c)
    div_step = torch.zeros(c.shape, dtype=torch.int32, device=device)

    for i in range(max_iter):
        # Места, где ещё не дивергировало
        mask = (div_step == 0)

        # Обновляем только живые точки
        z[mask] = z[mask] * z[mask] + c[mask]

        # Проверка выхода за пределы
        escaped = (torch.abs(z) > 2) & mask
        div_step[escaped] = i

    # Тем, кто не вышел — максимальная итерация
    div_step[div_step == 0] = max_iter

    return div_step.cpu()   # вернём на CPU для matplotlib


now = time.time()
image = mandelbrot_torch(x_min, x_max, y_min, y_max, width, height)
print("Torch GPU time:", time.time() - now)

plt.imshow(image, cmap='hot', extent=(x_min, x_max, y_min, y_max))
plt.colorbar(label="Iters before divergence")
plt.title("Mandelbrot fractal (PyTorch GPU)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()