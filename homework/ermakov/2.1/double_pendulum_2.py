import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from collections import deque

class DoublePendulum:
    def __init__(self, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.trail1 = deque(maxlen=500)
        self.trail2 = deque(maxlen=500)

    def Derivatives(self, t, state):
        theta1, w1, theta2, w2 = state

        delta_phi = theta1 - theta2
        cos_delta = np.cos(delta_phi)
        sin_delta = np.sin(delta_phi)
        sin_theta1 = np.sin(theta1)
        sin_theta2 = np.sin(theta2)

        L1 = self.L1
        L2 = self.L2
        g = self.g
    
        det = L1**2 * L2**2 * (1 + sin_delta**2)

        b1 = -L1 * L2 * w2**2 * sin_delta - 2 * g * L1 * sin_theta1
        b2 = L1 * L2 * w1**2 * sin_delta - g * L2 * sin_theta2

        acc1 = (b1 * L2**2 - b2 * L1 * L2 * cos_delta) / det
        acc2 = (b2 * 2 * L1**2 - b1 * L1 * L2 * cos_delta) / det
        
        return np.array((w1, acc1, w2, acc2))
    
    def GetPositions(self, theta1, theta2):
        x1 = self.L1 * np.sin(theta1)
        y1 = -self.L1 * np.cos(theta1)
        x2 = x1 + self.L2 * np.sin(theta2)
        y2 = y1 - self.L2 * np.cos(theta2)
        return x1, y1, x2, y2


def SimulatePendulum(pendulum, initial_conditions, t_span, dt):
    t_eval = np.arange(t_span[0], t_span[1], dt)

    return solve_ivp(
        pendulum.Derivatives,
        t_span,
        initial_conditions,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )


def CreateAnimation(pendulums, simulations, init_conds):
    assert len(pendulums) == len(simulations)

    #
    coords_params = []
    for i in range(len(pendulums)):
        theta1 = simulations[i].y[0]
        theta2 = simulations[i].y[2]
        t = simulations[i].t
        x1, y1, x2, y2 = np.array([pendulums[i].GetPositions(th1, th2) 
                            for th1, th2 in zip(theta1, theta2)]).T
        
        coords_params.append((theta1, theta2, t, x1, y1, x2, y2))
    #
    t = simulations[0].t

    fig = plt.figure(figsize=(15, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(pendulums)))
    
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Двойной маятник')
    #
    pendulums_trails = []
    for i in range(len(pendulums)):
        line, = ax1.plot([], [], 'o-', lw=2, color=colors[i], markersize=8, label=f'Pendulum: {i}; θ₁={init_conds[i][0]:.2f}, θ₂={init_conds[i][2]:.2f}, W₁={init_conds[i][1]}, W₂={init_conds[i][3]}')
        trail1, = ax1.plot([], [], '-', lw=1, alpha=0.3, color=colors[i])
        trail2, = ax1.plot([], [], '-', lw=1, alpha=0.3, color=colors[i])

        pendulums_trails.append((line, trail1, trail2))
    #
    ax1.legend()
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    
    ax2 = fig.add_subplot(222)
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Угол (рад)')
    ax2.set_title('Изменение угла θ₁')
    ax2.grid(True, alpha=0.3)
    #
    lines_theta1 = []
    for i in range(len(pendulums)):
        line, = ax2.plot([], [], '-', label=f'Pendulum: {i}', color=colors[i], lw=1.5)

        lines_theta1.append(line)
    #
    ax2.legend()
    ax2.set_xlim(0, simulations[0].t[-1])
    ax2.set_ylim(-2*np.pi, 2*np.pi)

    ax3 = fig.add_subplot(224)
    ax3.set_xlabel('Время (с)')
    ax3.set_ylabel('Угол (рад)')
    ax3.set_title('Изменение угла θ₂')
    ax3.grid(True, alpha=0.3)
    #
    lines_theta2 = []
    for i in range(len(pendulums)):
        line, = ax3.plot([], [], '-', label=f'Pendulum: {i}', color=colors[i], lw=1.5)

        lines_theta2.append(line)
    #
    ax3.legend()
    ax3.set_xlim(0, simulations[0].t[-1])
    ax3.set_ylim(-2*np.pi, 2*np.pi)
    
    def init():
        meta_array = []
        for i in range(len(pendulums_trails)):
            line, trail1, trail2 = pendulums_trails[i]
            line.set_data([], [])
            trail1.set_data([], [])
            trail2.set_data([], [])
            meta_array.extend([line, trail1, trail2])

            pendulums[i].trail1.clear()
            pendulums[i].trail2.clear()
        for i in range(len(lines_theta1)):
            line1 = lines_theta1[i]
            line1.set_data([], [])
            meta_array.extend([line1])
        for i in range(len(lines_theta2)):
            line2 = lines_theta2[i]
            line2.set_data([], [])
            meta_array.extend([line2])
        
        time_text.set_text('')
        meta_array.extend([time_text])
        return meta_array
    
    def animate(i):
        meta_array = []
        for j in range(len(pendulums)):
            theta1, theta2, t, x1, y1, x2, y2 = coords_params[j]
            thisx = [0, x1[i], x2[i]]
            thisy = [0, y1[i], y2[i]]

            line, trail1, trail2 = pendulums_trails[j]
            line.set_data(thisx, thisy)

            pendulums[j].trail1.append((x1[i], y1[i]))
            pendulums[j].trail2.append((x2[i], y2[i]))

            if len(pendulums[j].trail1) > 1:
                trail1_data = list(zip(*pendulums[j].trail1))
                trail1.set_data(trail1_data[0], trail1_data[1])
            
            if len(pendulums[j].trail2) > 1:
                trail2_data = list(zip(*pendulums[j].trail2))
                trail2.set_data(trail2_data[0], trail2_data[1])

            meta_array.extend([line, trail1, trail2])
        for j in range(len(pendulums)):
            theta1, theta2, t, x1, y1, x2, y2 = coords_params[j]
            line = lines_theta1[j]
            line.set_data(t[:i+1], theta1[:i+1])
            meta_array.extend([line])
        for j in range(len(pendulums)):
            theta1, theta2, t, x1, y1, x2, y2 = coords_params[j]
            line = lines_theta2[j]
            line.set_data(t[:i+1], theta2[:i+1])
            meta_array.extend([line])

        time_text.set_text(f'Время = {t[i]:.1f} с')    
        meta_array.extend([time_text])
        return meta_array
    
    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t), interval=25, blit=True
    )
    
    plt.tight_layout()
    return ani


if __name__ == "__main__":
    pendulum1 = DoublePendulum()
    pendulum2 = DoublePendulum()
    pendulum3 = DoublePendulum()
    pendulum4 = DoublePendulum()
    pendulum5 = DoublePendulum()

    t_max = 20
    dt = 0.03

    init_cond1 = [np.pi/3, 0.0,  np.pi/3, 0.0]
    simulation1 = SimulatePendulum(pendulum1, init_cond1, (0, t_max), dt)

    init_cond2 = [np.pi/3 + np.pi/36, 0.0,  np.pi/3+np.pi/36, 0.0]
    simulation2 = SimulatePendulum(pendulum2, init_cond2, (0, t_max), dt)

    init_cond3 = [np.pi/3 - np.pi/36, 0.0,  np.pi/3-np.pi/36, 0.0]
    simulation3 = SimulatePendulum(pendulum3, init_cond3, (0, t_max), dt)

    init_cond4 = [np.pi/3, 0.5, np.pi/3, 1.0]
    simulation4 = SimulatePendulum(pendulum4, init_cond4, (0, t_max), dt)

    init_cond5 = [np.pi/3, -0.5, np.pi/3, -1.0]
    simulation5 = SimulatePendulum(pendulum5, init_cond5, (0, t_max), dt)

    pendulums = [pendulum1, pendulum2, pendulum3, pendulum4, pendulum5]
    simulations = [simulation1, simulation2, simulation3, simulation4, simulation5]

    print("Максимальное Относительное отклонение θ₂ маятников от Pendulum 1")
    for i in range(1, len(pendulums)-1):
        theta2 = simulations[i].y[2]
        theta2_an = simulations[0].y[2]
        print(f"Pendulum {i+1}: {np.max(np.abs(theta2-theta2_an)) / np.max(np.abs(theta2_an))}")

    ani = CreateAnimation(
        pendulums, 
        simulations,
        [init_cond1, init_cond2, init_cond3, init_cond4, init_cond5],
    )
    plt.show()
