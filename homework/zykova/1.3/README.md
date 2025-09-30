# Решение домашнего задания 1.3: Вращение на орбите 2D

Программа для численного моделирования движения планет Солнечной системы с использованием RK4 и метода Эйлера

## Что делает программа

- Визуализирует орбиту Земли, Марса и Юпитера вокруг Солнца
- Сравнивает методы численного интегрирования (Эйлера и Рунге-Кутты 4-го порядка)
- Анализирует влияние Юпитера на орбиты Земли и Марса
- Визуализирует результаты с помощью анимаций и графиков

## Физические параметры

Используемые астрономические константы:
- **Единица длины**: 1 астрономическая единица (а.е.)
- **Единица времени**: 1 земной год
- **Единица массы**: масса Земли

Параметры планет:
| Планета | Большая полуось (а.е.) | Эксцентриситет | Период (лет) | Масса (отн. Земли) |
|---------|------------------------|----------------|--------------|-------------------|
| Земля   | 1.000                  | 0.017          | 1.000        | 1.000             |
| Марс    | 1.524                  | 0.094          | 1.881        | 0.107             |
| Юпитер  | 5.204                  | 0.049          | 11.862       | 317.800           |
| Солнце  | -                      | -              | -            | 332946.000        |

```python
a_Earth = 1.0 # большая полуось в а.е.
e_Earth = 0.017 # эксцентриситет орбиты
T_Earth = 1 # орбитальный период в годах
m_Earth = 1

a_Mars = 1.524
e_Mars = 0.094
T_Mars = 1.881
m_Mars = 0.107

a_Jupiter = 5.204
e_Jupiter = 0.049
T_Jupiter = 11.862
m_Jupiter = 317.8

m_Sun = 332.946
```

## Методы интегрирования
- Явный метод Эйлера

```python
def explicit_euler_step(s, dt, a, T):
    d1 = dt * d(s, a, T)
    return s + d1
```

- Метод Рунге-Кутты 4-го порядка (RK4)

```python
def rk4_step(s, dt, a, T):
    d1 = dt * d(s, a, T)
    d2 = dt * d(s + d1 / 2, a, T)
    d3 = dt * d(s + d2 / 2, a, T)
    d4 = dt * d(s + d3, a, T)
    return s + (d1 + 2 * d2 + 2 * d3 + d4) / 6
```

- Аналитическое решение - эталонные кеплеровские орбиты

```python
def analytic_solution(a, e):
    theta = np.linspace(0, 2*np.pi, 1000)
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y
```

## Задача трех тел

Чтобы оценить влияние Юпитера на Землю и Марс, решается задача трех тел с помошью RK4

```python
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

def init_cond_3body(a, e, m_star):
    mu = G * m_star
    r_peri = a * (1 - e)
    v_peri = np.sqrt(mu * (1 + e) / (a * (1 - e)))
    return r_peri, 0, 0, v_peri

def rk4_step_3N(s, dt, m1, m2, m3):
    k1 = dt * d_3N(s, m1, m2, m3)
    k2 = dt * d_3N(s + k1/2, m1, m2, m3)
    k3 = dt * d_3N(s + k2/2, m1, m2, m3)
    k4 = dt * d_3N(s + k3, m1, m2, m3)
    return s + (k1 + 2*k2 + 2*k3 + k4) / 6

def N3_task(a1, a2, e1, e2, m1, m2, m3, duration_years, N):
    x1, y1, vx1, vy1 = init_cond_3body(a1, e1, m3)
    x2, y2, vx2, vy2 = init_cond_3body(a2, e2, m3)
    x3, y3, vx3, vy3 = 0, 0, 0, 0
    time = np.linspace(0, duration_years, N+1)
    dt = time[1] - time[0]
    s = np.zeros((N+1, 12))
    s[0] = np.array([x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3])
    for i in range(1, N+1):
        s[i] = rk4_step_3N(s[i-1], dt, m1, m2, m3)
    return s, time

duration_years = 20
N = 5000

s1, time = N3_task(a_Earth, a_Jupiter, e_Earth, e_Jupiter, m_Earth, m_Jupiter, m_Sun, duration_years, N)
s2, time = N3_task(a_Mars, a_Jupiter, e_Mars, e_Jupiter, m_Mars, m_Jupiter, m_Sun, duration_years, N)

x_E = s1[:, 0] - s1[:, 4]   # Земля x относительно Солнца
y_E = s1[:, 1] - s1[:, 5]   # Земля y относительно Солнца
x_J = s1[:, 2] - s1[:, 4]   # Юпитер x относительно Солнца
y_J = s1[:, 3] - s1[:, 5]   # Юпитер y относительно Солнца
x_M = s2[:, 0] - s2[:, 4]   # Марс x относительно Солнца
y_M = s2[:, 1] - s2[:, 5]   # Марс y относительно Солнца

```





