# Projectile trajectory modeling using various integration methods
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PHYSICS

v0 = 100
alpha = math.radians(45.0)
g = 9.81
windage = 0.00

flight_range = v0 * v0 * math.sin(2 * alpha) / g
vx = v0 * math.cos(alpha)
vy = v0 * math.sin(alpha)

def d(s):
    x, y, vx, vy = s
    return np.array([vx, vy, -windage * vx, -g - windage * vy])

# NUMERIC

nsteps_an = 400
nsteps_num = 40000
dx = flight_range / nsteps_an
dt = dx / vx

# INTEGRATION_METHODS

def explicit_euler_step(s):
    d1 = dt * d(s)
    return s + d1

def implicit_euler_step(s):
    d1 = dt * d(s)
    d2 = dt * d(s + d1)
    return s + d2

def average_exp_imp_step(s):
    # RK2
    d1 = dt * d(s)
    d2 = dt * d(s + d1)
    return s + (d1 + d2) / 2

def half_point_step(s):
    # RK2
    d1 = dt * d(s)
    d2 = dt * d(s + d1 / 2)
    return s + d2

def rk4_step(s):
    # RK4
    d1 = dt * d(s)
    d2 = dt * d(s + d1 / 2)
    d3 = dt * d(s + d2 / 2)
    d4 = dt * d(s + d3)
    return s + (d1 + 2 * d2 + 2 * d3 + d4) / 6

# PREPROCESSING

# Initial state vector: [x, y, vx, vy]
s_0 = np.array([0.0, 0.0, vx, vy])

# Choose integration method - uncomment the one you want to use
# integration_method = explicit_euler_step
# integration_method = average_exp_imp_step
# integration_method = half_point_step
integration_method = rk4_step
method_name = "RK4"

# Analytical trajectory
x_an = np.linspace(0, flight_range, nsteps_an + 1)
y_an = x_an * math.tan(alpha) - g * x_an * x_an / (2.0 * v0 * v0 * math.cos(alpha) ** 2)

# Numerical trajectory arrays will be created in SIMULATION section

# SIMULATION

# Initialize arrays
s_trajectory = np.zeros((nsteps_num + 1, 4), dtype=np.float32)
s_trajectory[0] = s_0

# Integration loop
for i in range(1, nsteps_num + 1):
    s_trajectory[i] = integration_method(s_trajectory[i - 1])

    # Stop if projectile hits ground (but allow some initial points)
    if s_trajectory[i][1] < -1.0:  # Allow slight negative values
        s_trajectory[i][1] = 0.0  # Set to ground level
        break

# Extract x and y coordinates
x_num = s_trajectory[:, 0]
y_num = s_trajectory[:, 1]

# VISUALIZATION

# Find actual number of frames (where y >= 0)
max_frames = 0
for j in range(len(s_trajectory)):
    if s_trajectory[j][1] >= -1.0:  # Allow slight negative values
        max_frames = j + 1
    else:
        break

print(f"Total frames: {max_frames}")
print(f"Numerical trajectory points: {max_frames}")
print(f"Flight range (analytical): {max(x_an):.2f} m")
print(f"Flight range (numerical): {max(x_num[y_num > 0]):.2f} m")
print(f"Max height (analytical): {max(y_an):.2f} m")
print(f"Max height (numerical): {max(y_num):.2f} m")

# Create combined figure with two subplots
print("\nСоздание комбинированного графика...")
fig, (ax_static, ax_anim) = plt.subplots(1, 2, figsize=(15, 6))

# Left subplot: Static comparison
ax_static.axis('equal')
ax_static.set_title(f'Статическое сравнение\nАналитическая vs {method_name}')
ax_static.set_xlabel('X (м)')
ax_static.set_ylabel('Y (м)')
ax_static.grid(True, alpha=0.3)

ax_static.plot(x_an, y_an, 'b-', lw=3, label='Аналитическая (без сопротивления)')
ax_static.plot(x_num[:max_frames], y_num[:max_frames], 'r--', lw=3,
               label=f'Численная ({method_name})')

# Add ground level line
ax_static.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

ax_static.legend()
ax_static.set_xlim(0, max(x_an) * 1.1)
ax_static.set_ylim(0, max(max(y_an), max(y_num)) * 1.1)

# Right subplot: Animation
ax_anim.axis('equal')
ax_anim.set_title(f'Анимация траектории\n{method_name}')
ax_anim.set_xlabel('X (м)')
ax_anim.set_ylabel('Y (м)')
ax_anim.grid(True, alpha=0.3)

# Analytical trajectory (blue line)
traject_an = ax_anim.plot(x_an, y_an, 'b-', lw=2, label='Аналитическая')[0]

# Numerical trajectory (red line)
traject_num = ax_anim.plot([], [], 'r-', lw=2, label=f'Численная ({method_name})')[0]

# Add ground level line
ax_anim.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

ax_anim.legend()
ax_anim.set_xlim(0, max(x_an) * 1.1)
ax_anim.set_ylim(0, max(max(y_an), max(y_num)) * 1.1)

def init_anim():
    # Initialize animation subplot
    traject_an.set_data(x_an, y_an)
    traject_num.set_data([], [])
    return (traject_an, traject_num)

def loop_anim(i):
    # Show trajectory up to current frame
    show_idx = min(i + 1, max_frames)
    traject_num.set_data(x_num[:show_idx], y_num[:show_idx])
    return (traject_an, traject_num)

# Create animation with slower speed
ani = animation.FuncAnimation(
    fig=fig, func=loop_anim, init_func=init_anim,
    frames=max_frames, interval=50, repeat=False, blit=True
)

# Adjust layout and show combined plot
plt.tight_layout()
plt.show()
