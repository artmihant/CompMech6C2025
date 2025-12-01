"""
PINN для стационарного уравнения Бюргерса: u·u_x = ν·u_xx
Аналитическое решение: u(x) = -tanh(x / 2ν)

Это нелинейное ОДУ второго порядка — отличный пример для PINN!
Демонстрирует работу с производными второго порядка.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ============================================================================
# НЕЙРОСЕТЬ (та же архитектура)
# ============================================================================

class PINN(nn.Module):
    def __init__(self, hidden_layers=[64, 64, 64]):
        super().__init__()
        layers = []
        input_dim = 1
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.Tanh())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# ФИЗИКА: УРАВНЕНИЕ БЮРГЕРСА
# ============================================================================

def burgers_residual(model, x, nu):
    """
    Невязка стационарного уравнения Бюргерса: u·u_x - ν·u_xx = 0
    
    Здесь демонстрируется вычисление ВТОРОЙ производной через autograd!
    """
    x = x.requires_grad_(True)
    u = model(x)
    
    # Первая производная du/dx
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    # Вторая производная d²u/dx²
    # Дифференцируем u_x по x ещё раз
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]
    
    # Невязка: u·u_x - ν·u_xx = 0
    residual = u * u_x - nu * u_xx
    
    return residual


def compute_loss(model, x_interior, x_bc, u_bc, nu):
    """
    Loss = MSE(невязка) + λ·MSE(граничные условия)
    """
    # Physics loss
    res = burgers_residual(model, x_interior, nu)
    loss_physics = torch.mean(res**2)
    
    # Boundary loss
    u_pred = model(x_bc)
    loss_bc = torch.mean((u_pred - u_bc)**2)
    
    # Суммарный loss с весом для граничных условий
    loss = loss_physics + 100.0 * loss_bc
    
    return loss, loss_physics.item(), loss_bc.item()


# ============================================================================
# ОБУЧЕНИЕ
# ============================================================================

def train(model, x_int, x_bc, u_bc, nu, epochs=10000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    history = {'loss': [], 'physics': [], 'bc': []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, lp, lb = compute_loss(model, x_int, x_bc, u_bc, nu)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        history['loss'].append(loss.item())
        history['physics'].append(lp)
        history['bc'].append(lb)
        
        if (epoch + 1) % 2000 == 0:
            print(f"Epoch {epoch+1:5d} | Loss: {loss.item():.2e} | "
                  f"Phys: {lp:.2e} | BC: {lb:.2e}")
    
    return history


# ============================================================================
# ОСНОВНОЙ КОД
# ============================================================================

if __name__ == "__main__":
    # Параметры задачи
    nu = 0.1  # Вязкость (определяет ширину ударной волны)
    x_min, x_max = -3.0, 3.0
    
    # Аналитическое решение
    def exact_solution(x):
        return -np.tanh(x / (2 * nu))
    
    # Граничные условия: u(-3) ≈ 1, u(3) ≈ -1
    x_bc = torch.tensor([[-3.0], [3.0]])
    u_bc = torch.tensor([[exact_solution(-3.0)], [exact_solution(3.0)]])
    
    # Точки коллокации внутри области
    n_interior = 200
    x_interior = torch.linspace(x_min, x_max, n_interior).reshape(-1, 1)
    
    print("=" * 60)
    print("PINN для стационарного уравнения Бюргерса")
    print(f"Уравнение: u·u_x = ν·u_xx,  ν = {nu}")
    print(f"Решение:   u(x) = -tanh(x / 2ν)")
    print("=" * 60 + "\n")
    
    # Создаём и обучаем модель
    model = PINN(hidden_layers=[64, 64, 64])
    history = train(model, x_interior, x_bc, u_bc, nu, epochs=10000)
    
    # Визуализация
    x_plot = torch.linspace(x_min, x_max, 300).reshape(-1, 1)
    with torch.no_grad():
        u_pinn = model(x_plot).numpy()
    
    x_np = x_plot.numpy()
    u_exact = exact_solution(x_np)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Решение
    axes[0].plot(x_np, u_exact, 'b-', lw=2, label='Аналитическое')
    axes[0].plot(x_np, u_pinn, 'r--', lw=2, label='PINN')
    axes[0].axhline(0, color='gray', lw=0.5)
    axes[0].axvline(0, color='gray', lw=0.5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u')
    axes[0].set_title(f'Ударная волна Бюргерса (ν={nu})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Ошибка
    error = np.abs(u_pinn - u_exact)
    axes[1].semilogy(x_np, error, 'g-', lw=2)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('|ошибка|')
    axes[1].set_title(f'Макс. ошибка: {np.max(error):.2e}')
    axes[1].grid(True, alpha=0.3)
    
    # История
    axes[2].semilogy(history['loss'], label='Total')
    axes[2].semilogy(history['physics'], alpha=0.7, label='Physics')
    axes[2].semilogy(history['bc'], alpha=0.7, label='BC')
    axes[2].set_xlabel('Эпоха')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Обучение')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pinn_burgers.png', dpi=150)
    plt.show()
    
    print(f"\nМаксимальная ошибка: {np.max(error):.4e}")
    print(f"Средняя ошибка: {np.mean(error):.4e}")