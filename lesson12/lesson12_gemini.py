import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# Выбираем устройство (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

# Фиксируем seed для воспроизводимости
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Определение модели (Нейросеть)
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Простая полносвязная сеть (MLP)
        # Вход: 2 нейрона (x, t)
        # Скрытые слои: 4 слоя по 64 нейронов
        # Выход: 1 нейрон (u)
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # # Инициализация весов (Xavier) для лучшей сходимости
        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x, t):
        # Объединяем x и t в один тензор [N, 2]
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# ==========================================
# 2. Физика задачи (Вычисление невязки)
# ==========================================
def physics_loss(model, x, t, nu):
    """
    Вычисляет невязку уравнения Бюргерса:
    f = u_t + u*u_x - nu*u_xx
    Мы хотим, чтобы f -> 0
    """
    # Важно: разрешаем вычисление градиентов для входных данных
    x.requires_grad = True
    t.requires_grad = True
    
    u = model(x, t)
    
    # Вычисляем производные автоматически (Autograd)
    # create_graph=True нужен, чтобы потом взять вторую производную
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # Само уравнение (Residual)
    f = u_t + u * u_x - nu * u_xx
    
    return torch.mean(f ** 2) # Возвращаем MSE невязки

# ==========================================
# 3. Подготовка данных (Collocation points)
# ==========================================
def generate_data(num_boundary, num_collocation):
    """
    Генерирует точки для обучения:
    1. Граничные и начальные условия (Boundary + Initial)
    2. Точки коллокации внутри области (Physics)
    """
    
    # --- Граничные и Начальные условия (BC + IC) ---
    # Область: x ∈ [-1, 1], t ∈ [0, 1]
    
    # IC: t = 0, x ∈ [-1, 1] -> u = -sin(pi*x)
    x_ic = np.random.uniform(-1, 1, num_boundary).reshape(-1, 1)
    t_ic = np.zeros_like(x_ic)
    u_ic = -np.sin(np.pi * x_ic) # Аналитическое начальное условие
    
    # BC: x = -1, t ∈ [0, 1] -> u = 0
    t_bc1 = np.random.uniform(0, 1, num_boundary // 2).reshape(-1, 1)
    x_bc1 = -1 * np.ones_like(t_bc1)
    u_bc1 = np.zeros_like(t_bc1)
    
    # BC: x = 1, t ∈ [0, 1] -> u = 0
    t_bc2 = np.random.uniform(0, 1, num_boundary // 2).reshape(-1, 1)
    x_bc2 = np.ones_like(t_bc2)
    u_bc2 = np.zeros_like(t_bc2)
    
    # Собираем все точки, где мы знаем точное решение (границы)
    X_u_train = np.vstack([x_ic, x_bc1, x_bc2])
    T_u_train = np.vstack([t_ic, t_bc1, t_bc2])
    U_u_train = np.vstack([u_ic, u_bc1, u_bc2])
    
    # --- Точки коллокации (Collocation points) ---
    # Случайные точки внутри области, где мы будем проверять уравнение
    X_f_train = np.random.uniform(-1, 1, num_collocation).reshape(-1, 1)
    T_f_train = np.random.uniform(0, 1, num_collocation).reshape(-1, 1)
    
    return (torch.tensor(X_u_train, dtype=torch.float32).to(device),
            torch.tensor(T_u_train, dtype=torch.float32).to(device),
            torch.tensor(U_u_train, dtype=torch.float32).to(device),
            torch.tensor(X_f_train, dtype=torch.float32).to(device),
            torch.tensor(T_f_train, dtype=torch.float32).to(device))

# ==========================================
# 4. Обучение
# ==========================================

# Параметры задачи
NU = 0.01 / np.pi  # Вязкость
EPOCHS = 3000      # Количество итераций (для CPU ставим поменьше, но достаточно)
LR = 0.005         # Learning rate

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Генерируем данные
x_u, t_u, u_true, x_f, t_f = generate_data(num_boundary=500, num_collocation=5000)

print("Начинаем обучение PINN...")
start_time = time.time()

loss_history = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # 1. Потери на границах (Data Loss)
    # Нейросеть предсказывает u в граничных точках
    u_pred = model(x_u, t_u)
    loss_u = torch.mean((u_pred - u_true) ** 2)
    
    # 2. Потери физики (Physics Loss)
    # Нейросеть должна удовлетворять уравнению внутри области
    loss_f = physics_loss(model, x_f, t_f, NU)
    
    # Общая ошибка
    loss = loss_u + loss_f
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.5f} (Data: {loss_u.item():.5f}, Physics: {loss_f.item():.5f})")

print(f"Обучение завершено за {time.time() - start_time:.1f} сек.")

# ==========================================
# 5. Визуализация результатов
# ==========================================
plt.figure(figsize=(12, 5))

# График сходимости
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.yscale('log')
plt.title('Сходимость Loss функции')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')

# Визуализация решения u(x,t)
plt.subplot(1, 2, 2)
# Создаем сетку для отрисовки
x_plot = np.linspace(-1, 1, 100)
t_plot = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_plot, t_plot)

# Переводим в тензоры
X_tens = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
T_tens = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to(device)

# Получаем предсказание
with torch.no_grad():
    U_pred = model(X_tens, T_tens).cpu().numpy().reshape(100, 100)

cp = plt.contourf(T, X, U_pred, 100, cmap='jet')
plt.colorbar(cp)
plt.title('Предсказание PINN: u(x,t)')
plt.xlabel('t (время)')
plt.ylabel('x (координата)')
plt.show()

# Срез решения в разные моменты времени (сравнение динамики)
plt.figure(figsize=(10, 6))
t_snapshots = [0.0, 0.25, 0.5, 0.9]
colors = ['b', 'g', 'orange', 'r']

x_tens_slice = torch.tensor(x_plot[:, None], dtype=torch.float32).to(device)

for i, t_val in enumerate(t_snapshots):
    t_tens_slice = torch.ones_like(x_tens_slice) * t_val
    with torch.no_grad():
        u_slice = model(x_tens_slice, t_tens_slice).cpu().numpy()
    
    plt.plot(x_plot, u_slice, color=colors[i], label=f't = {t_val}', linewidth=2)

plt.title('Профили решения в разные моменты времени')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.grid(True)
plt.legend()
plt.show()