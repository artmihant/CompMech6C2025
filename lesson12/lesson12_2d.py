""" # Занятие 12 (Доп). PINN на сложной геометрии (2D) # """

r"""
В этом примере мы продемонстрируем главное преимущество нейросетевого подхода — **бессеточность (mesh-free)**.

### Проблема классических методов (FEM/FDM):
Для решения задачи на сложной фигуре (например, деталь двигателя) в классических методах необходимо:
1. Построить сетку (триангуляцию).
2. Следить за качеством элементов (чтобы не было слишком острых углов).
3. Перестраивать сетку при больших деформациях.

### Подход PINN:
Мы просто "набрасываем" облако точек внутрь области. Нейросети неважна связность точек, ей нужны только координаты $(x,y)$.

### Задача:
Решим уравнение Пуассона (стационарная теплопроводность с источником):
$$ -\Delta u = f(x,y) $$
$$ -(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}) = f(x,y) $$

В области $\Omega$ в виде **"Шестеренки"** (параметрически заданная кривая).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print(f"Device: {device}")


""" ## 1. Геометрия: Шестеренка (Gear) ## """

r"""
Определим область с помощью полярных координат.
Граница задается уравнением:
$$ R(\phi) = 1 + 0.2 \sin(8\phi) $$
Это создает фигуру с 8 "зубьями".
"""

def is_inside_gear(x, y):
    """
    Проверка, находится ли точка (x,y) внутри шестеренки.
    Используем полярные координаты.
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    # Параметрическая граница
    r_boundary = 1.0 + 0.2 * np.sin(8 * phi)
    
    # Точка внутри, если её радиус меньше радиуса границы
    return r < r_boundary

def generate_gear_points(n_points):
    """
    Генерация облака точек методом Rejection Sampling (Монте-Карло).
    Мы берем случайные точки в квадрате и оставляем только те, что внутри шестеренки.
    """
    points = []
    while len(points) < n_points:
        # Генерируем с запасом в квадрате [-1.5, 1.5]
        batch_x = np.random.uniform(-1.3, 1.3, n_points)
        batch_y = np.random.uniform(-1.3, 1.3, n_points)
        
        # Проверяем условие
        mask = is_inside_gear(batch_x, batch_y)
        
        # Добавляем хорошие точки
        valid_points = np.column_stack([batch_x[mask], batch_y[mask]])
        points.extend(valid_points)
        
    return np.array(points[:n_points])


def get_boundary_points(n_points):
    """Генерация точек строго на границе"""
    phi = np.random.uniform(0, 2*np.pi, n_points)
    r = 1.0 + 0.2 * np.sin(8 * phi)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y])

# 1. Точки внутри области (Collocation points) для уравнения
XY_collocation = generate_gear_points(6000)

x_col = torch.tensor(XY_collocation[:, 0:1], dtype=torch.float32).to(device)
y_col = torch.tensor(XY_collocation[:, 1:2], dtype=torch.float32).to(device)


# 2. Точки на границе (Boundary points) для граничных условий
XY_boundary = get_boundary_points(1000)

x_bnd = torch.tensor(XY_boundary[:, 0:1], dtype=torch.float32).to(device)
y_bnd = torch.tensor(XY_boundary[:, 1:2], dtype=torch.float32).to(device)


""" ## Визуализация облака точек ## """

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 7))
plt.scatter(XY_collocation[:, 0], XY_collocation[:, 1], s=7, color='dodgerblue', alpha=0.7, label='Collocation (внутри)')
plt.scatter(XY_boundary[:, 0], XY_boundary[:, 1], s=14, color='orangered', alpha=0.9, label='Boundary (граница)')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Точки: внутри области и на границе')
plt.legend()
plt.show()


""" ## 2. Точное решение (Method of Manufactured Solutions) ## """

r"""
Чтобы проверить точность метода, мы пойдем от обратного.
Зададим "истинное" решение:
$$ u_{exact} = \sin(3x) \cdot \cos(3y) $$

Тогда, подставив его в уравнение Пуассона $-\Delta u = f$, найдем правую часть $f$:
$$ \frac{\partial^2 u}{\partial x^2} = -9 \sin(3x)\cos(3y) = -9 u $$
$$ \frac{\partial^2 u}{\partial y^2} = -9 \sin(3x)\cos(3y) = -9 u $$
$$ f(x,y) = -(-9u - 9u) = 18 u $$

То есть мы будем решать уравнение:
$$ -\Delta u = 18 \sin(3x)\cos(3y) $$
с граничными условиями Дирихле, взятыми из точного решения.
"""

def exact_u(x, y):
    return np.sin(3*x) * np.cos(3*y)

def source_f(x, y):
    # f(x,y) для уравнения -Laplace(u) = f
    return 18 * np.sin(3*x) * np.cos(3*y)


""" ## 3. Нейросеть и Физика ## """

class PINN_2D(nn.Module):
    def __init__(self):
        super().__init__()
        # Вход: 2 координаты (x, y)
        # Выход: 1 температура (u)
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Xavier initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x, y):
        # Конкатенация координат
        inputs = torch.cat([x, y], dim=1)
        return self.net(inputs)

def poisson_loss(model, x, y):
    """
    Вычисление невязки уравнения Пуассона:
    Loss = mean( (-Laplace(u) - f)^2 )
    """
    x.requires_grad = True
    y.requires_grad = True
    
    u = model(x, y)
    
    # Первые производные
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    
    # Вторые производные
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    
    # Левая часть уравнения (-Laplace)
    lhs = -(u_xx + u_yy)
    
    # Правая часть (Source term)
    rhs = source_f(x, y)
    
    return torch.mean((lhs - rhs)**2)


""" ## 4. Подготовка данных и Обучение ## """



# Значения u на границе (берем из точного решения)
u_bnd_true = exact_u(x_bnd, y_bnd) # Здесь мы не используем .detach(), так как это константы

# Модель
model = PINN_2D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Начинаем обучение...")
start = time.time()
history = []

for epoch in range(5001):
    optimizer.zero_grad()
    
    # Loss на границе (BC)
    u_bnd_pred = model(x_bnd, y_bnd)
    loss_bc = torch.mean((u_bnd_pred - u_bnd_true)**2)
    
    # Loss внутри области (Physics)
    loss_phys = poisson_loss(model, x_col, y_col)
    
    # Суммарный Loss
    loss = loss_bc + loss_phys
    
    loss.backward()
    optimizer.step()
    
    history.append(loss.item())
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.5f} (BC: {loss_bc.item():.5f}, Phys: {loss_phys.item():.5f})")

print(f"Время обучения: {time.time() - start:.2f} сек")


""" ## 5. Визуализация ## """

# Подготовка сетки для рисования
# Мы создаем квадратную сетку, но рисуем только то, что внутри шестеренки
x_range = np.linspace(-1.3, 1.3, 200)
y_range = np.linspace(-1.3, 1.3, 200)
X, Y = np.meshgrid(x_range, y_range)

# Маска для скрытия внешних точек
mask = is_inside_gear(X, Y)
# Делаем NaN там, где точки снаружи, чтобы matplotlib не рисовал их
X_masked = np.where(mask, X, np.nan)
Y_masked = np.where(mask, Y, np.nan)

# Предсказание нейросети
x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32).to(device)

with torch.no_grad():
    u_pred_flat = model(x_flat, y_flat).cpu().numpy()

U_pred = u_pred_flat.reshape(X.shape)
U_pred_masked = np.where(mask, U_pred, np.nan)

# Точное решение
U_true_masked = np.where(mask, exact_u(X, Y), np.nan)
# Ошибка
Error_masked = np.abs(U_true_masked - U_pred_masked)

# Рисуем
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.contourf(X, Y, U_true_masked, levels=50, cmap='viridis')
plt.colorbar()
plt.title("Точное решение")
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.contourf(X, Y, U_pred_masked, levels=50, cmap='viridis')
plt.colorbar()
plt.title("Решение PINN")
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.contourf(X, Y, Error_masked, levels=50, cmap='inferno')
plt.colorbar()
plt.title("Абсолютная ошибка")
plt.axis('equal')

plt.tight_layout()
plt.show()

# Рисуем точки коллокации, чтобы показать "бессеточность"
plt.figure(figsize=(6, 6))
plt.scatter(XY_collocation[:,0], XY_collocation[:,1], s=1, alpha=0.5, label='Collocation')
plt.scatter(XY_boundary[:,0], XY_boundary[:,1], s=2, c='r', label='Boundary')
plt.title("Облако точек для обучения (Mesh-free)")
plt.axis('equal')
plt.legend()
plt.show()