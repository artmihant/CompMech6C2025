import numpy as np
import matplotlib.pyplot as plt
import helper as hh

# ----------------------
# Вспомогательные функции
# ----------------------
def norm(v):
    return np.linalg.norm(v)

def unit(v):
    n = norm(v)
    return v / n if n != 0 else v

def rotation_matrix_from_Omega_i_omega(Omega, i, omega):
    # Q = R_z(Omega) * R_x(i) * R_z(omega)
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(i), np.sin(i)
    co, so = np.cos(omega), np.sin(omega)
    RzO = np.array([[cO, -sO, 0],
                    [sO,  cO, 0],
                    [0,    0, 1]])
    RxI = np.array([[1,   0,    0],
                    [0,  ci, -si],
                    [0,  si,  ci]])
    RzW = np.array([[co, -so, 0],
                    [so,  co, 0],
                    [0,    0, 1]])
    return RzO @ RxI @ RzW

def kepler_E_from_M(M, e, tol=1e-12, maxiter=100):
    # Решение уравнения Кеплера E - e sin E = M для эллиптического случая e<1
    scalar = np.isscalar(M)
    Ms = np.atleast_1d(M)
    Es = np.empty_like(Ms, dtype=float)
    for idx, Mi in enumerate(Ms):
        # начальное приближение (популярные эвристики)
        if e < 0.8:
            E = Mi
        else:
            E = np.pi if Mi > np.pi else Mi  # грубая эвристика
        for _ in range(maxiter):
            f = E - e * np.sin(E) - Mi
            fp = 1 - e * np.cos(E)
            dE = - f / fp
            E += dE
            if abs(dE) < tol:
                break
        Es[idx] = E
    return Es[0] if scalar else Es

def kepler_F_from_M(M, e, tol=1e-12, maxiter=100):
    # Решение уравнения для гиперболы: e sinh F - F = M (e>1)
    scalar = np.isscalar(M)
    Ms = np.atleast_1d(M)
    Fs = np.empty_like(Ms, dtype=float)
    for idx, Mi in enumerate(Ms):
        # начальное приближение
        # для больших |M|: F ~ asinh(M/e)
        if Mi == 0.0:
            F = 0.0
        else:
            F = np.arcsinh(Mi / e)
        # улучшение: если M очень большой, дать чуть большее начальное приближение
        for _ in range(maxiter):
            f = e * np.sinh(F) - F - Mi
            fp = e * np.cosh(F) - 1.0
            dF = - f / fp
            F += dF
            if abs(dF) < tol:
                break
        Fs[idx] = F
    return Fs[0] if scalar else Fs

def solve_barker_for_D(A):
    # Решает кубик D + D^3/3 = A  -> эквивалент D^3 + 3D - 3A = 0
    # Используем явную формулу Кардано (один действительный корень)
    # Пусть p=3, q=-3A, уравнение x^3 + p x + q = 0 -> стандартно
    # Но проще воспользоваться np.cbrt для корней:
    # D = cbrt( (3A/2) + sqrt((3A/2)^2 + 1) ) - cbrt( - (3A/2) + sqrt((3A/2)^2 + 1) )
    C = 1.5 * A  # 3A/2
    sqrt_term = np.sqrt(C**2 + 1.0)
    # np.cbrt доступна; правильнее сохранять знак
    D = np.cbrt(C + sqrt_term) - np.cbrt(-C + sqrt_term)
    return D

def true_anomaly_from_time(ts, t_peri, e, a, p, mu):
    """
    ts: array of times
    возвращает массив истинных аномалий nu (в радиан) соответствующих ts.
    Поддерживает эллипсу (e<1), параболу (e==1), гиперболу (e>1).
    """
    ts = np.atleast_1d(ts)
    dt = ts - t_peri
    nu = np.empty_like(dt, dtype=float)

    if e < 1.0 - 1e-12:
        # эллиптический случай
        # среднее движение:
        n_mean = np.sqrt(mu / (a**3))
        M = n_mean * dt
        E = kepler_E_from_M(M, e)
        # перевод E -> nu:
        # tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)
        tan_half_nu = np.sqrt((1+e)/(1-e)) * np.tan(E/2.0)
        nu = 2.0 * np.arctan(tan_half_nu)
        # нормировка -π..π
        nu = (nu + np.pi) % (2*np.pi) - np.pi
    elif e > 1.0 + 1e-12:
        # гиперболический случай
        # a отрицателен при положительной eps -> a < 0
        # среднее "движение" для гиперболы:
        n_mean = np.sqrt(mu / (abs(a)**3))
        # Однако в уравнении M = e sinh F - F, M может быть положительным или отрицательным
        M = n_mean * dt
        # Решаем для F:
        F = kepler_F_from_M(M, e)
        # перевод F -> nu:
        # tan(nu/2) = sqrt((e+1)/(e-1)) * tanh(F/2)
        tanh_halfF = np.tanh(F/2.0)
        factor = np.sqrt((e+1.0)/(e-1.0))
        tan_half_nu = factor * tanh_halfF
        nu = 2.0 * np.arctan(tan_half_nu)
        # нормировка:
        nu = (nu + np.pi) % (2*np.pi) - np.pi
    else:
        # параболический случай (e == 1)
        # Barker: t - t_peri = 0.5 * sqrt(p^3/mu) * (D + D^3/3), D = tan(nu/2)
        # => A = (2*(t - t_peri)) / sqrt(p^3/mu)
        coef = 0.5 * np.sqrt(p**3 / mu)
        # избегаем деления на ноль
        A = (ts - t_peri) / coef  # так что A = 0.5*(...). мы инвертируем прямо в формуле ниже
        # из формулы: D + D^3/3 = 2*(t-t_peri)/sqrt(p^3/mu) = 2*dt / sqrt(...)
        A2 = 2.0 * (ts - t_peri) / np.sqrt(p**3 / mu)
        # решаем по одному
        for idx, Ai in enumerate(A2):
            D = solve_barker_for_D(Ai)
            nu[idx] = 2.0 * np.arctan(D)
        # нормировка
        nu = (nu + np.pi) % (2*np.pi) - np.pi

    return nu

# ----------------------
# Главная функция
# ----------------------
def two_body_analytic_trajectory(r1_0, v1_0, r2_0, v2_0, m1, m2, G, ex, ey , ez,
                                 t0, t1 , n_pts=1000, plot=True):
    M_total = m1 + m2
    mu = G * M_total  # гравитационный параметр (часто обозначают mu = G(M))
    # Относительные векторы и скорости
    r_rel_0 = r1_0 - r2_0
    v_rel_0 = v1_0 - v2_0
    r0 = r_rel_0.copy()
    v0 = v_rel_0.copy()
    r0_norm = norm(r0)
    v0_norm = norm(v0)

    # Энергия на единицу массы (специфическая энергия относительного движения)
    eps = 0.5 * v0_norm**2 - mu / r0_norm
    # большая полуось (если eps < 0 -> эллипс, =0 парабола, >0 гипербола)
    if abs(eps) < 1e-15:
        a = np.inf
    else:
        a = - mu / (2 * eps)   # обратите внимание: для гиперболы a < 0

    # Вектор углового момента (специфический)
    h = np.cross(r0, v0)
    h_norm = norm(h)

    # Эксцентриситет вектор (узнаём ориентацию перицентра)
    e_vec = (np.cross(v0, h) / mu) - (r0 / r0_norm)
    e = norm(e_vec)

    # Параметр орбиты p
    if np.isfinite(a):
        p = a * (1 - e**2)
    else:
        # парабола: p = h^2 / mu
        p = h_norm**2 / mu

    # Наклонение i
    if h_norm == 0.0:
        i = 0.0
    else:
        i = np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0))

    # Вектор узлов (node vector) n_vec = k x h
    kvec = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(kvec, h)
    n_norm = norm(n_vec)

    # Долгота восходящего узла Omega
    if n_norm < 1e-16:
        Omega = 0.0
    else:
        Omega = np.arctan2(n_vec[1], n_vec[0])  # atan2(ny, nx)
        Omega = (Omega + 2*np.pi) % (2*np.pi)

    # Аргумент перицентра omega
    # Для вырожденных случаев (круговая или экватор) omega может быть не определён.
    if e < 1e-12:
        # круговая орбита: аргумент перицентра не определён — поставим 0
        omega = 0.0
    else:
        if n_norm < 1e-16:
            # орбита лежит в экваториальной плоскости: измеряем omega от x-оси
            # проектируем e_vec на XY-плоскость (хотя h имеет нулевую z, этот случай уже попал сюда)
            omega = np.arctan2(e_vec[1], e_vec[0])
        else:
            # угол между n_vec и e_vec
            # using atan2 to get correct quadrant
            num = np.dot(np.cross(n_vec, e_vec), h) / h_norm
            den = np.dot(n_vec, e_vec)
            omega = np.arctan2(num, den)

    # Истинная аномалия nu0 (угол между e_vec и r0)
    if e < 1e-16:
        # круговая орбита: истинная аномалия определяется от направления n_vec (или x-axis в орб.пл.)
        # проще: определяем nu0 в плоскости XY проекцией r0 на орбитальную систему координат
        # но для унификации просто вычислим угол между х-осью и проекцией r0 в орб.пл.
        # Сделаем временно Q=I для этого определения (величина лишь для нахождения времени перицентра)
        nu0 = np.arctan2(r0[1], r0[0])
    else:
        num = np.dot(np.cross(e_vec, r0), h) / h_norm
        den = np.dot(e_vec, r0)
        nu0 = np.arctan2(num, den)

    # Вычислим момент прохождения перицентра t_peri
    if e < 1.0 - 1e-12:
        # эллипс
        # Эксцентриская аномалия E0
        cosE0 = (e + np.cos(nu0)) / (1 + e * np.cos(nu0))
        cosE0 = np.clip(cosE0, -1.0, 1.0)
        E0 = np.arccos(cosE0)
        if np.sin(nu0) < 0:
            E0 = -E0
        M0 = E0 - e * np.sin(E0)
        n_mean = np.sqrt(mu / (a**3))
        t_peri = t0 - M0 / n_mean
    elif e > 1.0 + 1e-12:
        # гипербола
        # найдем начальную гиперболическую аномалию F0 из nu0:
        # tan(nu/2) = sqrt((e+1)/(e-1)) * tanh(F/2) => обратное: tanh(F/2) = tan(nu/2)/factor
        factor = np.sqrt((e+1.0)/(e-1.0))
        tan_half_nu0 = np.tan(nu0/2.0)
        tanh_halfF0 = tan_half_nu0 / factor
        # чтобы избежать выхода за пределы: clip tanh_halfF0 to (-1+eps,1-eps)
        tanh_halfF0 = np.clip(tanh_halfF0, -1+1e-12, 1-1e-12)
        F0 = 2.0 * np.arctanh(tanh_halfF0)
        M0 = e * np.sinh(F0) - F0
        n_mean = np.sqrt(mu / (abs(a)**3))  # положительная величина
        t_peri = t0 - M0 / n_mean
    else:
        # парабола (e ~= 1)
        # получаем D0 = tan(nu0/2)
        D0 = np.tan(nu0/2.0)
        # Barker: t - t_peri = 0.5*sqrt(p^3/mu)*(D + D^3/3)
        t_peri = t0 - 0.5 * np.sqrt(p**3 / mu) * (D0 + D0**3 / 3.0)

    # Матрица поворота из орбитальной плоскости в инерциальную
    Q = rotation_matrix_from_Omega_i_omega(Omega, i, omega)

    # Временная сетка
    ts = np.linspace(t0, t1, n_pts)

    # Находим истинную аномалию nu для всех ts (унифицированно)
    nu_array = true_anomaly_from_time(ts, t_peri, e, a, p, mu)

    # Радиус в орбитальной плоскости: r = p / (1 + e cos nu)
    # Здесь p>0 (по определению p = h^2/mu для всех типов)
    r_orb = p / (1.0 + e * np.cos(nu_array))

    # координаты в орбитальной плоскости
    x_orb = r_orb * np.cos(nu_array)
    y_orb = r_orb * np.sin(nu_array)
    zeros = np.zeros_like(x_orb)

    # собираем r_rel(t) в инерциальной системе: r = Q @ [x_orb, y_orb, 0]
    r_rel = np.stack([x_orb, y_orb, zeros], axis=0)  # shape (3, N)
    r_inertial = (Q @ r_rel).T  # shape (N,3)

    # центр масс линейно движется (с постоянной скоростью)
    Rcm_0 = (m1 * r1_0 + m2 * r2_0) / M_total
    Vcm = (m1 * v1_0 + m2 * v2_0) / M_total
    Rcm_t = Rcm_0 + np.outer(ts - t0, Vcm)  # shape (N,3)

    # позиции тел
    r1_t = Rcm_t + ( m2 / M_total) * r_inertial
    r2_t = Rcm_t + (-m1 / M_total) * r_inertial

    # Проекция в плоскость движения CM: нормаль = Vcm
    Vcm_norm = norm(Vcm)
    if Vcm_norm < 1e-14:
        # если CM неподвижен, можно взять любую нормаль (напр. z)
        n_plane = np.array([0.0, 0.0, 1.0])
    else:
        n_plane = Vcm / Vcm_norm

    # проекции
    r1_rel_cm = r1_t - Rcm_t
    r2_rel_cm = r2_t - Rcm_t
    x1 = r1_rel_cm @ ex
    y1 = r1_rel_cm @ ey
    x2 = r2_rel_cm @ ex
    y2 = r2_rel_cm @ ey

    # определим тип орбиты
    if e < 1.0 - 1e-12:
        orbit_type = 'elliptic'
    elif e > 1.0 + 1e-12:
        orbit_type = 'hyperbolic'
    else:
        orbit_type = 'parabolic'

    result = {
        't': ts,
        'r1': r1_t, 'r2': r2_t,
        'r_rel': r_inertial,
        'Rcm': Rcm_t, 'Vcm': Vcm,
        'orbit_elements': {
            'a': a, 'e': e, 'p': p, 'i': i, 'Omega': Omega, 'omega': omega,
            'h': h, 'e_vec': e_vec, 'type': orbit_type, 't_peri': t_peri
        },
        'proj': {
            'ex': ex, 'ey': ey, 'n_plane': n_plane,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
        }
    }

    if plot:
        plt.figure(figsize=(8,8))
        plt.plot(x1, y1, label='Тело 1 (проекция в плоскости CM)')
        plt.plot(x2, y2, label='Тело 2 (проекция в плоскости CM)')
        plt.plot([0], [0], 'ko', label='Центр масс (проекция)')
        plt.axis('equal')
        plt.xlabel('x (в плоскости CM)')
        plt.ylabel('y (в плоскости CM)')
        plt.title(f'Аналитическое решение задачи двух тел (проекция на плоскость CM) — {orbit_type}')
        plt.legend()
        plt.grid(True)
        plt.show()

    return result
