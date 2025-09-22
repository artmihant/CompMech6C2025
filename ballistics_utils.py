"""
Вспомогательные функции для визуализации баллистических траекторий
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def plot_trajectories(theta_list, integrate_func, x_target, y_target, x_start, y_start,
                     title="Траектории для разных углов стрельбы", custom_labels=None):
    """
    Визуализация траекторий для списка углов

    Args:
        theta_list: список углов в радианах
        integrate_func: функция интегрирования траектории
        x_target, y_target: координаты цели
        x_start, y_start: координаты старта
        title: заголовок графика
        custom_labels: пользовательские метки для легенд (если None, используются углы)
    """
    plt.figure(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(theta_list)))

    for i, theta in enumerate(theta_list):
        final_state, trajectory = integrate_func(theta, return_full_trajectory=True)

        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]

        # Определяем метку для легенды
        if custom_labels is not None and i < len(custom_labels):
            label = custom_labels[i]
        else:
            label = f'{np.degrees(theta):.1f}°'

        plt.plot(x_coords, y_coords, color=colors[i], label=label)

    # Целевая точка
    plt.scatter([x_target], [y_target], color='red', s=100, marker='x',
               label=f'Цель ({x_target:.1f}, {y_target:.1f})', zorder=5)

    # Стартовая точка
    plt.scatter([x_start], [y_start], color='green', s=100, marker='o',
               label=f'Старт ({x_start:.1f}, {y_start:.1f})', zorder=5)

    plt.xlabel('Расстояние, м')
    plt.ylabel('Высота, м')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_convergence(theta_history, residual_history, method_name):
    """
    Визуализация процесса сходимости метода

    Args:
        theta_history: история изменения угла
        residual_history: история изменения невязки
        method_name: название метода для заголовка
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    iterations = range(len(theta_history))

    # График изменения угла
    ax1.plot(iterations, np.degrees(theta_history), 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Угол, градусы')
    ax1.set_title(f'Сходимость угла ({method_name})')
    ax1.grid(True, alpha=0.3)

    # График изменения невязки
    ax2.plot(iterations, residual_history, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Итерация')
    ax2.set_ylabel('Невязка, м')
    ax2.set_title(f'Сходимость невязки ({method_name})')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Импорт функций из основного модуля (нужно будет настроить)
# from ballistics import integrate_trajectory, x_target, y_target, x_start, y_start
