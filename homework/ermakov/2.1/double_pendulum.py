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


def SimulatePendulum(pendulum, theta1, w1, theta2, w2, t_span, dt):
    initial_conditions = [theta1, w1, theta2, w2]
    t_eval = np.arange(t_span[0], t_span[1], dt)

    return solve_ivp(
        pendulum.Derivatives,
        t_span,
        initial_conditions,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8
    )


def CreateAnimation(pendulum, simulation):
    theta1 = simulation.y[0]
    theta2 = simulation.y[2]
    t = simulation.t
    
    positions = np.array([pendulum.GetPositions(th1, th2) 
                          for th1, th2 in zip(theta1, theta2)])
    x1, y1, x2, y2 = positions.T
 
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Двойной маятник')
    line, = ax1.plot([], [], 'o-', lw=2, color='black', markersize=8)
    trail1, = ax1.plot([], [], '-', lw=1, alpha=0.3, color='blue')
    trail2, = ax1.plot([], [], '-', lw=1, alpha=0.3, color='red')
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Угол (рад)')
    ax2.set_title('Изменение углов')
    ax2.grid(True, alpha=0.3)
    line1, = ax2.plot([], [], 'b-', label='θ₁', lw=1.5)
    line2, = ax2.plot([], [], 'r-', label='θ₂', lw=1.5)
    ax2.legend()
    ax2.set_xlim(0, simulation.t[-1])
    ax2.set_ylim(min(min(theta1), min(theta2))-0.5, 
                 max(max(theta1), max(theta2))+0.5)
    
    def init():
        line.set_data([], [])
        trail1.set_data([], [])
        trail2.set_data([], [])
        line1.set_data([], [])
        line2.set_data([], [])
        time_text.set_text('')
        pendulum.trail1.clear()
        pendulum.trail2.clear()
        return line, trail1, trail2, time_text, line1, line2
    
    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        line.set_data(thisx, thisy)
        
        pendulum.trail1.append((x1[i], y1[i]))
        pendulum.trail2.append((x2[i], y2[i]))
        
        if len(pendulum.trail1) > 1:
            trail1_data = list(zip(*pendulum.trail1))
            trail1.set_data(trail1_data[0], trail1_data[1])
        
        if len(pendulum.trail2) > 1:
            trail2_data = list(zip(*pendulum.trail2))
            trail2.set_data(trail2_data[0], trail2_data[1])
        
        time_text.set_text(f'Время = {t[i]:.1f} с')
        
        line1.set_data(t[:i+1], theta1[:i+1])
        line2.set_data(t[:i+1], theta2[:i+1])
        
        return line, trail1, trail2, time_text, line1, line2
    
    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t), interval=25, blit=True
    )
    
    plt.tight_layout()
    return ani


if __name__ == "__main__":
    pendulum = DoublePendulum()

    theta1 = np.pi/18
    w1 = 0.0
    theta2 = np.pi/18
    w2 = 0.0

    t_max = 20
    dt = 0.03

    simulation = SimulatePendulum(pendulum, theta1, w1, theta2, w2, (0, t_max), dt)

    ani = CreateAnimation(pendulum, simulation)
    plt.show()
