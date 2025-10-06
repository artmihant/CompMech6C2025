import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks


def HastingsPowellModel(t, state, r, K, a1, b1, a2, b2, d1, d2):
    """
    Модель Hastings-Powell для трехвидовой пищевой цепи
    
    dX/dt = r*X*(1 - X/K) - a1*X*Y/(1 + b1*X)
    dY/dt = a1*X*Y/(1 + b1*X) - a2*Y*Z/(1 + b2*Y) - d1*Y
    dZ/dt = a2*Y*Z/(1 + b2*Y) - d2*Z
    
    X - жертва, Y - хищник, Z - суперхищник
    """
    X, Y, Z = state
    
    # Функциональные отклики
    f1 = a1 * X / (1 + b1 * X)  # Тип II функциональный отклик для Y
    f2 = a2 * Y / (1 + b2 * Y)  # Тип II функциональный отклик для Z
    
    # Система дифференциальных уравнений
    dX_dt = r * X * (1 - X/K) - f1 * Y
    dY_dt = f1 * Y - f2 * Z - d1 * Y
    dZ_dt = f2 * Z - d2 * Z
    
    return [dX_dt, dY_dt, dZ_dt]

# Установка параметров модели
params = {
    'r': 1.0,      # скорость роста жертвы
    'K': 1.0,      # емкость среды для жертвы
    'a1': 5.0,     # максимальная скорость потребления жертвы хищником
    'b1': 3.0,     # константа полунасыщения для хищника
    'a2': 0.1,     # максимальная скорость потребления хищника суперхищником
    'b2': 2.0,     # константа полунасыщения для суперхищника
    'd1': 0.4,     # смертность хищника
    'd2': 0.01     # смертность суперхищника
}

# Начальные условия
initial_conditions = [0.5, 0.1, 0.1]  # [X0, Y0, Z0]

# Временной интервал
t_span = (0, 10000)
t_eval = np.linspace(0, 10000, 50000)

solution = solve_ivp(
    HastingsPowellModel,
    t_span,
    initial_conditions,
    args=tuple(params.values()),
    method='RK45',
    t_eval=t_eval,
    dense_output=True,
    rtol=1e-9,
    atol=1e-12
)

t = solution.t
X, Y, Z = solution.y

fig = plt.figure(figsize=(16, 12))

# 1. Временные ряды популяций
ax1 = fig.add_subplot(3, 3, (1, 4))
ax1.plot(t, X, 'b-', label='X (жертва)', linewidth=0.8)
ax1.plot(t, Y, 'r-', label='Y (хищник)', linewidth=0.8)
ax1.plot(t, Z, 'g-', label='Z (суперхищник)', linewidth=0.8)
ax1.set_xlabel('Время')
ax1.set_ylabel('Плотность популяции')
ax1.set_title('Динамика популяций во времени')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Увеличенный фрагмент временных рядов (последние 1000 единиц времени)
ax2 = fig.add_subplot(3, 3, 7)
idx_zoom = t > 9000
ax2.plot(t[idx_zoom], X[idx_zoom], 'b-', label='X', linewidth=1)
ax2.plot(t[idx_zoom], Y[idx_zoom], 'r-', label='Y', linewidth=1)
ax2.plot(t[idx_zoom], Z[idx_zoom], 'g-', label='Z', linewidth=1)
ax2.set_xlabel('Время')
ax2.set_ylabel('Плотность популяции')
ax2.set_title('Динамика популяций (увеличение)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Фазовый портрет X-Y
ax3 = fig.add_subplot(3, 3, 2)
ax3.plot(X[idx_zoom], Y[idx_zoom], 'b-', linewidth=0.5, alpha=0.7)
ax3.plot(X[idx_zoom][0], Y[idx_zoom][0], 'ro', markersize=5, label='Начало')
ax3.set_xlabel('X (жертва)')
ax3.set_ylabel('Y (хищник)')
ax3.set_title('Фазовый портрет X-Y')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Фазовый портрет Y-Z
ax4 = fig.add_subplot(3, 3, 5)
ax4.plot(Y[idx_zoom], Z[idx_zoom], 'r-', linewidth=0.5, alpha=0.7)
ax4.plot(Y[idx_zoom][0], Z[idx_zoom][0], 'ro', markersize=5, label='Начало')
ax4.set_xlabel('Y (хищник)')
ax4.set_ylabel('Z (суперхищник)')
ax4.set_title('Фазовый портрет Y-Z')
ax4.grid(True, alpha=0.3)
ax4.legend()

# 5. Фазовый портрет X-Z
ax5 = fig.add_subplot(3, 3, 8)
ax5.plot(X[idx_zoom], Z[idx_zoom], 'g-', linewidth=0.5, alpha=0.7)
ax5.plot(X[idx_zoom][0], Z[idx_zoom][0], 'ro', markersize=5, label='Начало')
ax5.set_xlabel('X (жертва)')
ax5.set_ylabel('Z (суперхищник)')
ax5.set_title('Фазовый портрет X-Z')
ax5.grid(True, alpha=0.3)
ax5.legend()

# 6. 3D фазовый портрет
ax6 = fig.add_subplot(3, 3, (3, 9), projection='3d')
ax6.plot(X[idx_zoom], Y[idx_zoom], Z[idx_zoom], 'b-', linewidth=0.5, alpha=0.7)
ax6.plot([X[idx_zoom][0]], [Y[idx_zoom][0]], [Z[idx_zoom][0]], 'ro', markersize=5)
ax6.set_xlabel('X (жертва)')
ax6.set_ylabel('Y (хищник)')
ax6.set_zlabel('Z (суперхищник)')
ax6.set_title('3D фазовый портрет')
ax6.view_init(elev=20, azim=45)

plt.suptitle('Модель Hastings-Powell: Трехвидовая пищевая цепь', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Анализ периодичности
def AnalyzePeriodicity(t, X, Y, Z):
    print("\n=== Анализ динамики системы ===\n")
    
    # Используем последнюю часть временного ряда для анализа
    idx = t > 8000
    t_analysis = t[idx]
    X_analysis = X[idx]
    Y_analysis = Y[idx]
    Z_analysis = Z[idx]
    
    for name, data in [('X (жертва)', X_analysis), 
                       ('Y (хищник)', Y_analysis), 
                       ('Z (суперхищник)', Z_analysis)]:
        peaks, _ = find_peaks(data, distance=50)
        if len(peaks) > 1:
            periods = np.diff(t_analysis[peaks])
            mean_period = np.mean(periods)
            std_period = np.std(periods)
            print(f"{name}:")
            print(f"  Средний период: {mean_period:.2f} ± {std_period:.2f}")
            print(f"  Количество циклов: {len(peaks)}")
            print(f"  Диапазон значений: [{np.min(data):.4f}, {np.max(data):.4f}]")
            print()

    print("Характер динамики:")
    if np.std(periods) / mean_period > 0.1:
        print("  Система демонстрирует хаотическое поведение")
    else:
        print("  Система демонстрирует периодические колебания")

AnalyzePeriodicity(t, X, Y, Z)
