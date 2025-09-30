# Решение домашнего задания 2.2: Задача трех тел

## Что делает программа

Визуализирует решения задачи трех тел с помощью анимации

## Методы интегрирования

- Метод Рунге-Кутты 4-го порядка (RK4) с адаптивным выбором шага интегрирования

```python

# Функция правых частей систему ОДУ
def d_3N(s, m1, m2, m3):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = s
    f12x, f12y = gravitational_force(m1, m2, x1, y1, x2, y2)
    f13x, f13y = gravitational_force(m1, m3, x1, y1, x3, y3)
    f23x, f23y = gravitational_force(m2, m3, x2, y2, x3, y3)
    ax1 = (f12x + f13x) / m1
    ay1 = (f12y + f13y) / m1
    ax2 = (-f12x + f23x) / m2
    ay2 = (-f12y + f23y) / m2
    ax3 = (-f13x - f23x) / m3
    ay3 = (-f13y - f23y) / m3
    return np.array([vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3])

#Шаг метода RK4
def rk4_step_3N(s, dt, m1, m2, m3):
    k1 = dt * d_3N(s, m1, m2, m3)
    k2 = dt * d_3N(s + k1/2, m1, m2, m3)
    k3 = dt * d_3N(s + k2/2, m1, m2, m3)
    k4 = dt * d_3N(s + k3, m1, m2, m3)
    return s + (k1 + 2*k2 + 2*k3 + k4) / 6

# Оценка шага по времени для стабильного интегрирования
def estimate_stable_timestep(s, m1, m2, m3, safety_factor=0.01):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = s
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    r_min = min(r12, r13, r23)
    if r_min < 1e-8:
        return 1e-6
    total_mass = m1 + m2 + m3
    orbital_period_estimate = 2 * np.pi * np.sqrt(r_min**3 / (G * total_mass))
    dt_stable = safety_factor * orbital_period_estimate
    return min(dt_stable, 0.1)

# Будем отслеживать энергию для проверки точности
def total_energy(s, m1, m2, m3):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = s
    kinetic = 0.5 * (m1*(vx1**2+vy1**2) + m2*(vx2**2+vy2**2) + m3*(vx3**2+vy3**2))
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    potential = -G * (m1*m2/r12 + m1*m3/r13 + m2*m3/r23)
    return kinetic + potential

def adaptive_N3_task(m1, m2, m3, x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3, duration, max_points=10000):
    s_current = np.array([x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3])
    dt = estimate_stable_timestep(s_current, m1, m2, m3)
    results = [s_current.copy()]
    times = [0.0]
    energies = [total_energy(s_current, m1, m2, m3)]
    t_current = 0.0
    point_count = 1
    while t_current < duration and point_count < max_points:
        s_prev = s_current.copy()
        s_try = rk4_step_3N(s_current, dt, m1, m2, m3)
        r12 = np.sqrt((s_try[2]-s_try[0])**2 + (s_try[3]-s_try[1])**2)
        r13 = np.sqrt((s_try[4]-s_try[0])**2 + (s_try[5]-s_try[1])**2)
        r23 = np.sqrt((s_try[4]-s_try[2])**2 + (s_try[5]-s_try[3])**2)
        min_distance = min(r12, r13, r23)
        if min_distance < 1e-10 or np.any(np.isnan(s_try)) or np.any(np.isinf(s_try)):
            dt *= 0.5
            continue
        s_current = s_try
        t_current += dt
        
        if point_count % 10 == 0 or t_current - times[-1] > duration/100:
            results.append(s_current.copy())
            times.append(t_current)
            energies.append(total_energy(s_current, m1, m2, m3))
            point_count += 1
        
        new_dt = estimate_stable_timestep(s_current, m1, m2, m3)
        dt = min(new_dt, duration - t_current)
    
    return np.array(results), np.array(times), np.array(energies)

```
## Периодические решения

Для демонстрации возможностей программы были выбраны были выбраны такие массы тел и начальные условия, что полученные орбиты обладают свойством периодичности.

### Фигура в виде восьмерки

```python
m1_1, m2_1, m3_1 = 1.0, 1.0, 1.0
x1_1, y1_1 = -0.97000436, 0.24308753
x2_1, y2_1 = 0.97000436, -0.24308753
x3_1, y3_1 = 0.0, 0.0
vx1_1, vy1_1 = 0.4662036850, 0.4323657300
vx2_1, vy2_1 = 0.4662036850, 0.4323657300
vx3_1, vy3_1 = -0.93240737, -0.86473146
```

### Решение Лагранжа

Начальные точки - стороны равностороннего треугольника. 

Для построения графика переходим в систему отсчета центра масс системы трех тел.

Почему такие начальные скорости:

Для каждого тела центробежная сила должна уравновешивать сумму гравитационных сил от двух других тел:
F_центробежная = F_гравитация_12 + F_гравитация_13
F_центробежная = m * ω² * R
(Для равностороннего треугольника расстояние до центра масс R = side / √3)
F_гравитация = F = G * m * m / side² (для одного тела)
Векторная сумма двух сил под углом 60°:
F_сумма = 2 * F * cos(30°) = 2 * F * (√3/2) = F * √3

В итоге:
Условие равновесия: m * ω² * R = G * m * m / side² * √3
m * ω² * (side / √3) = G * m² / side² * √3
ω² * (side / √3) = G * m / side² * √3
ω² * side = G * m / side² * 3
ω² = G * 3m / side³
При m = 1.0: ω = √(G * 3.0 / side³)

```python
m1_2, m2_2, m3_2 = 1.0, 1.0, 1.0
side = 2.0
x1_2, y1_2 = 0.0, 0.0
x2_2, y2_2 = side, 0.0
x3_2, y3_2 = side/2, side * math.sqrt(3)/2

cm_x = (m1_2*x1_2 + m2_2*x2_2 + m3_2*x3_2) / (m1_2 + m2_2 + m3_2)
cm_y = (m1_2*y1_2 + m2_2*y2_2 + m3_2*y3_2) / (m1_2 + m2_2 + m3_2)

x1_2 -= cm_x; y1_2 -= cm_y
x2_2 -= cm_x; y2_2 -= cm_y
x3_2 -= cm_x; y3_2 -= cm_y

omega = math.sqrt(G * 3.0 / side**3)
vx1_2, vy1_2 = -omega * y1_2, omega * x1_2
vx2_2, vy2_2 = -omega * y2_2, omega * x2_2
vx3_2, vy3_2 = -omega * y3_2, omega * x3_2
```

### Две тяжелые звезды и одна легкая планета

Для построения графика переходим в систему отсчета центра масс системы трех тел.

```python
m1_3, m2_3, m3_3 = 1.0, 1.0, 0.01
distance = 1.0
x1_3, y1_3 = -distance/2, 0.0
x2_3, y2_3 = distance/2, 0.0
x3_3, y3_3 = 0.0, 2.5

total_mass = m1_3 + m2_3 + m3_3
cm_x3 = (m1_3*x1_3 + m2_3*x2_3 + m3_3*x3_3) / total_mass
cm_y3 = (m1_3*y1_3 + m2_3*y2_3 + m3_3*y3_3) / total_mass
x1_3 -= cm_x3; y1_3 -= cm_y3
x2_3 -= cm_x3; y2_3 -= cm_y3
x3_3 -= cm_x3; y3_3 -= cm_y3

v = math.sqrt(G * (m1_3 + m2_3) / (4 * distance))
vx1_3, vy1_3 = 0.0, v
vx2_3, vy2_3 = 0.0, -v
dx = x3_3 - (m1_3*x1_3 + m2_3*x2_3) / (m1_3 + m2_3)
dy = y3_3 - (m1_3*y1_3 + m2_3*y2_3) / (m1_3 + m2_3)
distance = math.sqrt(dx**2 + dy**2)
v_orbital = math.sqrt(G * (m1_3 + m2_3) / distance)
vx3_3 = -v_orbital * dy / distance
vy3_3 = v_orbital * dx / distance
```


