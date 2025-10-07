import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

sigma = 10.0
rho = 28.0
beta = 8.0/3.0

def LorenzSystem(t, state):
    """
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Начальные условия
x0_1 = [1.0, 1.0, 1.0]  # Первая траектория
x0_2 = [1.001, 1.0, 1.0]  # Вторая траектория

t_span = (0, 50)
t_eval = np.linspace(0, 50, 10000)

sol1 = solve_ivp(LorenzSystem, t_span, x0_1, t_eval=t_eval, method='RK45', rtol=1e-8)
sol2 = solve_ivp(LorenzSystem, t_span, x0_2, t_eval=t_eval, method='RK45', rtol=1e-8)

# Извлечение решений
x1, y1, z1 = sol1.y
x2, y2, z2 = sol2.y
t = sol1.t

# Создание фигуры с подграфиками
fig = plt.figure(figsize=(16, 12))

# 1. 3D траектория (для визуализации)
ax1 = fig.add_subplot(3, 3, 1, projection='3d')
ax1.plot(x1, y1, z1, 'b-', linewidth=0.5, alpha=0.7)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D траектория аттрактора Лоренца')
ax1.view_init(elev=20, azim=45)

# 2. Фазовый портрет в плоскости XY
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(x1, y1, 'b-', linewidth=0.5, alpha=0.7)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Проекция на плоскость XY')
ax2.grid(True, alpha=0.3)

# 3. Фазовый портрет в плоскости XZ
ax3 = fig.add_subplot(3, 3, 3)
ax3.plot(x1, z1, 'b-', linewidth=0.5, alpha=0.7)
ax3.set_xlabel('X')
ax3.set_ylabel('Z')
ax3.set_title('Проекция на плоскость XZ')
ax3.grid(True, alpha=0.3)

# 4. Фазовый портрет в плоскости YZ
ax4 = fig.add_subplot(3, 3, 4)
ax4.plot(y1, z1, 'b-', linewidth=0.5, alpha=0.7)
ax4.set_xlabel('Y')
ax4.set_ylabel('Z')
ax4.set_title('Проекция на плоскость YZ')
ax4.grid(True, alpha=0.3)

# 5. Координата X от времени
ax5 = fig.add_subplot(3, 3, 5)
ax5.plot(t, x1, 'b-', linewidth=0.8)
ax5.set_xlabel('Время t')
ax5.set_ylabel('X(t)')
ax5.set_title('Координата X от времени')
ax5.grid(True, alpha=0.3)

# 6. Координата Y от времени
ax6 = fig.add_subplot(3, 3, 6)
ax6.plot(t, y1, 'g-', linewidth=0.8)
ax6.set_xlabel('Время t')
ax6.set_ylabel('Y(t)')
ax6.set_title('Координата Y от времени')
ax6.grid(True, alpha=0.3)

# 7. Координата Z от времени
ax7 = fig.add_subplot(3, 3, 7)
ax7.plot(t, z1, 'r-', linewidth=0.8)
ax7.set_xlabel('Время t')
ax7.set_ylabel('Z(t)')
ax7.set_title('Координата Z от времени')
ax7.grid(True, alpha=0.3)

# 8. Сравнение двух траекторий (чувствительность к начальным условиям)
ax8 = fig.add_subplot(3, 3, 8)
ax8.plot(t, x1, 'b-', label=f'x₀ = {x0_1[0]}', linewidth=1)
ax8.plot(t, x2, 'r--', label=f'x₀ = {x0_2[0]}', linewidth=1)
ax8.set_xlabel('Время t')
ax8.set_ylabel('X(t)')
ax8.set_title('Чувствительность к начальным условиям (X)')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Расхождение траекторий
ax9 = fig.add_subplot(3, 3, 9)
distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
ax9.semilogy(t, distance, 'k-', linewidth=1.5)
ax9.set_xlabel('Время t')
ax9.set_ylabel('Расстояние между траекториями')
ax9.set_title('Расхождение траекторий (логарифмическая шкала)')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Создание отдельной фигуры для детального сравнения траекторий
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

# Сравнение в 3D (две траектории)
ax_3d = fig2.add_subplot(221, projection='3d')
ax_3d.plot(x1[:5000], y1[:5000], z1[:5000], 'b-', linewidth=0.8, alpha=0.7, label='Траектория 1')
ax_3d.plot(x2[:5000], y2[:5000], z2[:5000], 'r-', linewidth=0.8, alpha=0.7, label='Траектория 2')
ax_3d.scatter(x0_1[0], x0_1[1], x0_1[2], c='b', s=50, marker='o')
ax_3d.scatter(x0_2[0], x0_2[1], x0_2[2], c='r', s=50, marker='o')
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.set_title('Сравнение двух траекторий в 3D')
ax_3d.legend()

# Проекция XY - сравнение
axes[0, 1].plot(x1, y1, 'b-', linewidth=0.5, alpha=0.5, label='Траектория 1')
axes[0, 1].plot(x2, y2, 'r-', linewidth=0.5, alpha=0.5, label='Траектория 2')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')
axes[0, 1].set_title('Сравнение в плоскости XY')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Проекция XZ - сравнение
axes[1, 0].plot(x1, z1, 'b-', linewidth=0.5, alpha=0.5, label='Траектория 1')
axes[1, 0].plot(x2, z2, 'r-', linewidth=0.5, alpha=0.5, label='Траектория 2')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Z')
axes[1, 0].set_title('Сравнение в плоскости XZ')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Фазовое пространство скоростей
dx1 = sigma * (y1 - x1)
dx2 = sigma * (y2 - x2)
axes[1, 1].plot(x1, dx1, 'b-', linewidth=0.5, alpha=0.5, label='Траектория 1')
axes[1, 1].plot(x2, dx2, 'r-', linewidth=0.5, alpha=0.5, label='Траектория 2')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('dX/dt')
axes[1, 1].set_title('Фазовый портрет X vs dX/dt')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
