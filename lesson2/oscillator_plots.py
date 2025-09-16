import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def create_comparison_plots(t, data_list, title="Сравнение методов"):
    """
    Создает графики сравнения методов интегрирования для осциллятора

    Args:
        t: массив времени
        data_list: список словарей с данными методов
                каждый словарь содержит: 'label', 'color', 'u', 'v', 'e'
        title: заголовок фигуры

    Returns:
        fig, axes: фигура и массив осей matplotlib
    """

    for data in data_list:
        print(f"\nНачальная энергия (метод {data['label']}): {data['e'][0]} Дж")
        print(f"Конечная энергия (метод {data['label']}): {data['e'][-1]} Дж")
        print(f"Изменение энергии: {data['e'][-1] - data['e'][0]} Дж")
        print(f"Относительное изменение: {((data['e'][-1] - data['e'][0])/data['e'][0]*100)}%")


    # Создаем фигуру с подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)

    # 1. Смещение как функция времени
    for data in data_list:
        linestyle = '--' if 'Аналитическое' in data['label'] else '-'
        axes[0, 0].plot(t, data['u'], label=data['label'], linewidth=2,
                    color=data['color'], linestyle=linestyle)
    axes[0, 0].set_xlabel('Время, с')
    axes[0, 0].set_ylabel('Смещение, м')
    axes[0, 0].set_title('Смещение осциллятора')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Фазовый портрет
    for data in data_list:
        linestyle = '--' if 'Аналитическое' in data['label'] else '-'
        axes[0, 1].plot(data['u'], data['v'], label=data['label'], linewidth=2,
                    color=data['color'], linestyle=linestyle)
    axes[0, 1].set_xlabel('Смещение, м')
    axes[0, 1].set_ylabel('Скорость, м/с')
    axes[0, 1].set_title('Фазовый портрет')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Энергия
    for data in data_list:
        linestyle = '--' if 'Аналитическое' in data['label'] else '-'
        axes[1, 0].plot(t, data['e'], label=data['label'], linewidth=2,
                    color=data['color'], linestyle=linestyle)
    axes[1, 0].set_xlabel('Время, с')
    axes[1, 0].set_ylabel('Энергия, Дж')
    axes[1, 0].set_title('Полная энергия')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Пустой subplot для анимации
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Анимация осциллятора')

    plt.tight_layout()
    return fig, axes



def plots(t, T, plot_data):

    # Создаем графики
    fig, axes = create_comparison_plots(t, plot_data, 'Сравнение методов')

    # 4. Анимация пружинного осциллятора для всех трех методов в четвертом subplot

    # Настройка четвертого subplot для анимации
    ax_anim = axes[1, 1]
    # Устанавливаем пределы - пружина будет от -1.5 до 1.5, с запасом
    max_displacement = 1.5
    ax_anim.set_xlim(-max_displacement, max_displacement)
    ax_anim.set_ylim(-0.5, 0.5)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True, alpha=0.3)
    ax_anim.set_title('Анимация пружинных осцилляторов')

    # Фиксированная стенка слева
    ax_anim.plot([-max_displacement, -max_displacement], [-0.2, 0.2], 'k-', lw=4)
    ax_anim.text(-max_displacement, 0.3, 'Стенка', ha='center', va='bottom')

    # Линии для каждого осциллятора (пружины + масса)
    spring_lines = []
    mass_lines = []

    # Используем цвета из plot_data
    for data in plot_data:
        # Пружина (будет состоять из нескольких сегментов)
        spring_line, = ax_anim.plot([], [], '-', lw=2, color=data['color'], label=data['label'])
        # Масса (квадратик на конце пружины)
        mass_line, = ax_anim.plot([], [], 's-', markersize=12, color=data['color'], markerfacecolor=data['color'])
        spring_lines.append(spring_line)
        mass_lines.append(mass_line)

    ax_anim.legend()

    # Текст для энергии
    energy_text = ax_anim.text(0.02, 0.98, '', transform=ax_anim.transAxes,
                            verticalalignment='top', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def animate_pendulums(frame):
        """Функция анимации для обновления пружинных осцилляторов в subplot"""
        energy_info = []
        methods_data = [(data['u'], data['v'], data['label']) for data in plot_data]

        for i, (x, v, label) in enumerate(methods_data):
            # Позиция массы
            mass_x = x[frame]
            mass_y = 0  # масса движется горизонтально

            # Создаем пружину как серию зигзагов
            wall_x = -max_displacement  # позиция стенки
            spring_length = mass_x - wall_x
            n_coils = 8  # количество витков пружины

            # Создаем точки пружины
            spring_points_x = []
            spring_points_y = []

            for j in range(n_coils * 2 + 1):
                # Распределяем точки вдоль пружины
                fraction = j / (n_coils * 2)
                x_pos = wall_x + fraction * spring_length

                # Зигзаг пружины
                if j % 2 == 0:
                    y_pos = 0  # центральная линия
                else:
                    y_pos = 0.1 * (-1)**(j//2)  # зигзаг вверх-вниз

                spring_points_x.append(x_pos)
                spring_points_y.append(y_pos)

            # Добавляем точку массы
            spring_points_x.append(mass_x)
            spring_points_y.append(mass_y)

            # Обновляем пружину
            spring_lines[i].set_data(spring_points_x, spring_points_y)

            # Обновляем массу
            mass_lines[i].set_data([mass_x], [mass_y])

        energy_text.set_text('\n'.join(energy_info))

        return spring_lines + mass_lines + [energy_text]

    def init_animation():
        """Инициализация анимации"""
        for spring_line in spring_lines:
            spring_line.set_data([], [])
        for mass_line in mass_lines:
            mass_line.set_data([], [])
        energy_text.set_text('')
        return spring_lines + mass_lines + [energy_text]

    # Создаем анимацию в subplot
    print("\nСоздание анимации в subplot...")

    # Создание анимации для subplot
    fps = 20
    interval = 1000 / fps  # интервал в миллисекундах
    min_length = min(*[len(data['u']) for data in plot_data])
    frames = np.arange(0, min_length, max(1, min_length//(fps*T)))  # выбираем кадры для плавности

    anim = animation.FuncAnimation(
        fig, animate_pendulums, init_func=init_animation,
        frames=len(frames), interval=interval, blit=True
    )

    plt.tight_layout()
    plt.show()
