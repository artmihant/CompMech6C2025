import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')


class OrbitalSimulation:
    def __init__(self, r0, v0, GM=1.0, dt=0.01, t_max=100):
        self.r0 = np.array(r0, dtype=float)
        self.v0 = np.array(v0, dtype=float)
        self.GM = GM
        self.dt = dt
        self.t_max = t_max
        self.n_steps = int(t_max / dt)
  
    def Acceleration(self, r):
        r_mag = np.linalg.norm(r)
        return -self.GM * r / r_mag**3
    
    def EulerMethod(self):
        r = np.zeros((self.n_steps, 2))
        v = np.zeros((self.n_steps, 2))
        
        r[0] = self.r0
        v[0] = self.v0
        
        for i in range(1, self.n_steps):
            a = self.Acceleration(r[i-1])
            v[i] = v[i-1] + a * self.dt
            r[i] = r[i-1] + v[i-1] * self.dt
            
        return r, v
    
    def RK2Method(self):
        positions = np.zeros((self.n_steps, 2))
        velocities = np.zeros((self.n_steps, 2))
        
        positions[0] = self.r0
        velocities[0] = self.v0
        
        for i in range(1, self.n_steps):
            r = positions[i-1]
            v = velocities[i-1]
            
            # k1
            k1_v = self.Acceleration(r) * self.dt
            k1_r = v * self.dt
            
            # k2
            k2_v = self.Acceleration(r + k1_r) * self.dt
            k2_r = (v + k1_v) * self.dt

            velocities[i] = v + (k1_v + k2_v) / 2
            positions[i] = r + (k1_r + k2_r) / 2
            
        return positions, velocities
    

    def RK4Method(self):
        positions = np.zeros((self.n_steps, 2))
        velocities = np.zeros((self.n_steps, 2))
        
        positions[0] = self.r0
        velocities[0] = self.v0
        
        for i in range(1, self.n_steps):
            r = positions[i-1]
            v = velocities[i-1]
            
            # k1
            k1_v = self.Acceleration(r) * self.dt
            k1_r = v * self.dt
            
            # k2
            k2_v = self.Acceleration(r + k1_r/2) * self.dt
            k2_r = (v + k1_v/2) * self.dt
            
            # k3
            k3_v = self.Acceleration(r + k2_r/2) * self.dt
            k3_r = (v + k2_v/2) * self.dt
            
            # k4
            k4_v = self.Acceleration(r + k3_r) * self.dt
            k4_r = (v + k3_v) * self.dt
            
            velocities[i] = v + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
            positions[i] = r + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6
            
        return positions, velocities
    
    def CalcuteEnergy(self, r, v):
        kinetic = 0.5 * np.sum(v**2, axis=1)
        potential = -self.GM / np.linalg.norm(r, axis=1)
        return kinetic + potential
    
    def CalculateAngularMomentum(self, r, v):
        # L = r × v
        return r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]
    
    def AnalyticalOrbit(self):
        r0_mag = np.linalg.norm(self.r0)
        v0_mag = np.linalg.norm(self.v0)
        
        E = 0.5 * v0_mag**2 - self.GM / r0_mag
        L = self.r0[0] * self.v0[1] - self.r0[1] * self.v0[0]
        
        a = -self.GM / (2 * E)
        e = np.sqrt(1 + 2 * E * L**2 / self.GM**2)
        
        time = np.linspace(0, 2*np.pi, 6000)
        r = a * (1 - e**2) / (1 + e * np.cos(time))
        
        x = r * np.cos(time)
        y = r * np.sin(time)
        
        return x, y, a, e
    
    def AnimateOrbits(self, speed_factor=10):
        # Calc trajectory
        r_euler, v_euler = self.EulerMethod()
        r_rk2, v_rk2 = self.RK2Method()
        r_rk4, v_rk4 = self.RK4Method()
        x_analytical, y_analytical, a, e = self.AnalyticalOrbit()
        
        # energy and angular moment
        energy_euler = self.CalcuteEnergy(r_euler, v_euler)
        energy_rk2 = self.CalcuteEnergy(r_rk2, v_rk2)
        energy_rk4 = self.CalcuteEnergy(r_rk4, v_rk4)
        angular_momentum_euler = self.CalculateAngularMomentum(r_euler, v_euler)
        angular_momentum_rk2 = self.CalculateAngularMomentum(r_rk2, v_rk2)
        angular_momentum_rk4 = self.CalculateAngularMomentum(r_rk4, v_rk4)
        
        # animation
        time = np.linspace(0, self.t_max, self.n_steps)
        
        fig = plt.figure(figsize=(16, 10))
        
        # Орбиты
        ax1 = plt.subplot(2, 2, 1)
        # Устанавливаем пределы осей, учитывая все три метода
        x_min = min(np.min(r_euler[:,0]), np.min(r_rk2[:,0]), np.min(r_rk4[:,0]))
        x_max = max(np.max(r_euler[:,0]), np.max(r_rk2[:,0]), np.max(r_rk4[:,0]))
        y_min = min(np.min(r_euler[:,1]), np.min(r_rk2[:,1]), np.min(r_rk4[:,1]))
        y_max = max(np.max(r_euler[:,1]), np.max(r_rk2[:,1]), np.max(r_rk4[:,1]))
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Орбиты')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        ax1.plot(x_analytical, y_analytical, 'g--', alpha=0.3, label='Аналитическая')
        
        sun = Circle((0, 0), 0.05, color='yellow', zorder=10)
        ax1.add_patch(sun)
        
        # Линии траекторий
        line_euler, = ax1.plot([], [], 'b-', alpha=0.7, label='Эйлер', linewidth=1)
        line_rk2, = ax1.plot([], [], 'orange', alpha=0.7, label='RK2', linewidth=1)
        line_rk4, = ax1.plot([], [], 'r-', alpha=0.7, label='RK4', linewidth=1)
        
        # Точки-планеты
        point_euler, = ax1.plot([], [], 'bo', markersize=8)
        point_rk2, = ax1.plot([], [], 'o', color='orange', markersize=8)
        point_rk4, = ax1.plot([], [], 'ro', markersize=8)
        
        ax1.legend(loc='upper right')
        
        # График энергии
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_xlim(0, self.t_max)
        ax2.set_ylim(min(min(energy_euler), min(energy_rk2), min(energy_rk4))*1.1,
                     max(max(energy_euler), max(energy_rk2), max(energy_rk4))*1.1)
        ax2.set_xlabel('Время')
        ax2.set_ylabel('Полная энергия')
        ax2.set_title('Сохранение энергии')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=energy_euler[0], color='g', linestyle='--', alpha=0.3, label='Начальная')
        
        line_energy_euler, = ax2.plot([], [], 'b-', label='Эйлер')
        line_energy_rk2, = ax2.plot([], [], 'orange', label='RK2')
        line_energy_rk4, = ax2.plot([], [], 'r-', label='RK4')
        ax2.legend(loc='upper right')
        
        # График углового момента
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_xlim(0, self.t_max)
        ax3.set_xlabel('Время')
        ax3.set_ylabel('Угловой момент')
        ax3.set_title('Сохранение углового момента')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=angular_momentum_euler[0], color='g', linestyle='--', alpha=0.3, label='Начальный')
        
        line_L_euler, = ax3.plot([], [], 'b-', label='Эйлер')
        line_L_rk2, = ax3.plot([], [], 'g-', label='RK2')
        line_L_rk4, = ax3.plot([], [], 'r-', label='RK4')
        ax3.legend(loc='upper right')
        
        # Текстовая информация
        ax6 = plt.subplot(2, 2, 4)
        ax6.axis('off')
        
        info_text = ax6.text(0.1, 0.9, '', transform=ax6.transAxes, 
                            fontsize=10, verticalalignment='top',
                            fontfamily='monospace')
        
        plt.suptitle(f'Анимация орбитального движения\n(dt={self.dt})', 
                    fontsize=14, fontweight='bold')
        
        def init():
            line_euler.set_data([], [])
            line_rk2.set_data([], [])
            line_rk4.set_data([], [])
            point_euler.set_data([], [])
            point_rk2.set_data([], [])
            point_rk4.set_data([], [])

            line_energy_euler.set_data([], [])
            line_energy_rk2.set_data([], [])
            line_energy_rk4.set_data([], [])
            line_L_euler.set_data([], [])
            line_L_rk2.set_data([], [])
            line_L_rk4.set_data([], [])
            info_text.set_text('')

            return (line_euler, line_rk2, line_rk4, point_euler, point_rk2, point_rk4,
                   line_energy_euler, line_energy_rk2, line_energy_rk4,
                   line_L_euler, line_L_rk2, line_L_rk4, info_text)
        
        def animate(frame):
            idx = frame * speed_factor
            if idx >= self.n_steps:
                idx = self.n_steps - 1
            
            line_euler.set_data(r_euler[:idx, 0], r_euler[:idx, 1])
            line_rk2.set_data(r_rk2[:idx, 0], r_rk2[:idx, 1])
            line_rk4.set_data(r_rk4[:idx, 0], r_rk4[:idx, 1])
            point_euler.set_data([r_euler[idx, 0]], [r_euler[idx, 1]])
            point_rk2.set_data([r_rk2[idx, 0]], [r_rk2[idx, 1]])
            point_rk4.set_data([r_rk4[idx, 0]], [r_rk4[idx, 1]])
            
            line_energy_euler.set_data(time[:idx], energy_euler[:idx])
            line_energy_rk2.set_data(time[:idx], energy_rk2[:idx])
            line_energy_rk4.set_data(time[:idx], energy_rk4[:idx])
            line_L_euler.set_data(time[:idx], angular_momentum_euler[:idx])
            line_L_rk2.set_data(time[:idx], angular_momentum_rk2[:idx])
            line_L_rk4.set_data(time[:idx], angular_momentum_rk4[:idx])
            
            if idx > 0:
                rel_error_energy_euler = np.abs(energy_euler[:idx] - energy_euler[0]) / np.max(np.abs(energy_euler))
                rel_error_energy_rk2 = np.abs(energy_rk2[:idx] - energy_rk2[0]) / np.max(np.abs(energy_rk2))
                rel_error_energy_rk4 = np.abs(energy_rk4[:idx] - energy_rk4[0]) / np.max(np.abs(energy_rk4))

                x_rk4, y_rk4 = r_rk4[idx][0], r_rk4[idx][1]
                x_rk2, y_rk2 = r_rk2[idx][0], r_rk2[idx][1]
                x_euler, y_euler = r_euler[idx][0], r_euler[idx][1]

                a_half_axis, b_half_axis = (np.max(x_analytical)+np.abs(np.min(x_analytical)))/2.0, np.max(y_analytical)
                x_center = (np.max(x_analytical)+np.min(x_analytical)) / 2.0

                okolo_odin_euler = ((x_euler-x_center)*(x_euler-x_center)) / (a_half_axis*a_half_axis) + (y_euler*y_euler) / (b_half_axis*b_half_axis)
                okolo_odin_rk2 = ((x_rk2-x_center)*(x_rk2-x_center)) / (a_half_axis*a_half_axis) + (y_rk2*y_rk2) / (b_half_axis*b_half_axis)
                okolo_odin_rk4 = ((x_rk4-x_center)*(x_rk4-x_center)) / (a_half_axis*a_half_axis) + (y_rk4*y_rk4) / (b_half_axis*b_half_axis)
            
            info_str = f"Время: {time[idx]:.2f}\n"
            info_str += f"Большая полуось: {a:.3f}\n"
            info_str += f"Эксцентриситет: {e:.3f}\n"
            info_str += f"{'Метод':<10} {'Энергия':<12} {'Угл. момент':<12}\n"
            info_str += f"{'='*35}\n"
            info_str += f"{'Начальные':<10} {energy_euler[0]:>11.4f} {angular_momentum_euler[0]:>11.4f}\n"
            info_str += f"{'Эйлер':<10} {energy_euler[idx]:>11.4f} {angular_momentum_euler[idx]:>11.4f}\n"
            info_str += f"{'RK2':<10} {energy_rk2[idx]:>11.4f} {angular_momentum_rk2[idx]:>11.4f}\n"
            info_str += f"{'RK4':<10} {energy_rk4[idx]:>11.4f} {angular_momentum_rk4[idx]:>11.4f}\n"
            
            if idx > 0:
                info_str += f"\nОтносительные ошибки:\n"
                info_str += f"{'Эйлер E:':<10} {rel_error_energy_euler[-1]:.2e}\n"
                info_str += f"{'RK2 E:':<10} {rel_error_energy_rk2[-1]:.2e}\n"
                info_str += f"{'RK4 E:':<10} {rel_error_energy_rk4[-1]:.2e}\n"
                info_str += f"{'Эйлер Орб:':<10} {np.abs(1-okolo_odin_euler):.3e}\n"
                info_str += f"{'RK2 Орб:':<10} {np.abs(1-okolo_odin_rk2):.3e}\n"
                info_str += f"{'RK4 Орб:':<10} {np.abs(1-okolo_odin_rk4):.2e}\n"
            
            info_text.set_text(info_str)
            
            return (line_euler, line_rk2, line_rk4, point_euler, point_rk2, point_rk4,
                   line_energy_euler, line_energy_rk2, line_energy_rk4,
                   line_L_euler, line_L_rk2, line_L_rk4, info_text)
        
        frames = self.n_steps // speed_factor
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=frames, interval=20, 
                           blit=True, repeat=True)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    r0 = [1.0, 0.0] 
    v0 = [0.0, 1.2]

    simulation = OrbitalSimulation(r0, v0, GM=1.0, dt=0.01, t_max=60)
    simulation.AnimateOrbits(speed_factor=20)
