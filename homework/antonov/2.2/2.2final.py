"""
# Задача трёх тел

Выполнил: Антонов Андрей, студент 626 группы.

## Аннотация

Данная работа посвящена численному моделированию и анализу классической задачи трёх тел в двумерной постановке. В качестве основного объекта исследования выбрана периодическая устойчивая конфигурация «восьмёрка» (решение Ченсинера–Монтгомери).

В рамках исследования произведена визуализация движения тел и проведен анализ динамических характеристик системы. Численное интегрирование уравнений движения выполнено с использованием четырех методов: метода Рунге-Кутты 4-го порядка (RK4), явного и неявного методов Адамса 4-го порядка, а также симплектического интегратора Верлета.

В ходе выполнения задания были построены графики модулей скоростей и фазовых траекторий. Особое внимание уделено анализу устойчивости решений: вычислены ошибки сохранения энергии и углового момента на больших временных интервалах. Дополнительно реализован расчет показателя Ляпунова, подтверждающий устойчивый характер рассматриваемой конфигурации при малых возмущениях.

Результаты эксперимента показывают, что методы Верлета, РК4, неявный Адамс является наиболее эффективным с точки зрения точности соблюдения законов сохранения. Реализованная визуализация позволяет наглядно оценить эволюцию системы во времени. Рассчитан показатель Ляпунова, подтверждающий отсутствие хаотического режима в системе при заданных параметрах системы.

## Ключевые слова

Задача трёх тел, метод Рунге-Кутты, явный метод Адамса, неявный метод Адамса, симплектический интегратор Верлета.

## 1. Введение

История задачи восходит к трудам Исаака Ньютона ("Математические начала натуральной философии", 1687 год). Если задача двух тел была полностью решена Ньютоном (законы Кеплера), то добавление третьего тела делало систему аналитически неразрешимой в общем виде [1] [2]. На протяжении XVIII и XIX веков величайшие математики — Эйлер, Лагранж, Якоби — искали частные решения. Были найдены точки либрации (точки Лагранжа) и некоторые специфические конфигурации, однако общее решение оставалось недостижимым.

Переломным моментом стала работа Анри Пуанкаре в конце XIX века. Исследуя ограниченную задачу трёх тел, он доказал, что система неинтегрируема в смысле Лиувилля: для неё не существует достаточного количества аналитических интегралов движения. Пуанкаре показал, что даже малые изменения начальных условий могут приводить к кардинально различным траекториям. Это открытие положило начало теории детерминированного хаоса.

Особый интерес для данного исследования представляет частное периодическое решение типа «восьмёрка». Впервые численно оно было обнаружено Кристофером Муром в 1993 году, а строгое математическое доказательство его существования дали Ален Ченсинер и Ричард Монтгомери в 2000 году. Это уникальная «хореографическая» орбита, где три тела одинаковой массы движутся по одной замкнутой кривой, следуя друг за другом. Именно эта конфигурация выбрана в данной работе в качестве базовой модели.

Ввиду отсутствия общего аналитического решения, основным инструментом исследования задачи трёх тел стало численное моделирование. В данной работе рассматриваются три класса методов:

- Метод Рунге-Кутты (RK4) [3]. Разработанный немецкими математиками Карлом Рунге и Мартином Куттой на рубеже XIX–XX веков. Он обеспечивает высокую точность при сравнительной простоте реализации, так как не требует вычисления производных высших порядков, заменяя их усреднением наклона функции в нескольких точках интервала.

- Методы Адамса [4]. Эти методы восходят к работам британского астронома Джона Куча Адамса и Фрэнсиса Бэшфорта. В отличие от одношаговых методов Рунге-Кутты, методы Адамса являются многошаговыми: для расчета следующей точки они используют «предысторию» движения. В работе исследуются как явный метод (Адамса-Бэшфорта), так и неявный (Адамса-Моултона) 4-го порядка. Последний, хоть и требует итерационного решения уравнений на каждом шаге, обычно обладает лучшими свойствами устойчивости.

- Метод Верлета [5], [6]. Широкую известность под именем «метод Верле» алгоритм получил после работы Лу Верле 1967 года в области молекулярной динамики. Его ключевое преимущество — симплектичность. Это геометрическое свойство позволяет методу сохранять фазовый объем и, как следствие, удерживать полную энергию консервативной системы ограниченной даже на очень больших временных интервалах, что критически важно для задачи N-тел.

Особый интерес представляет исследование устойчивости и хаотичности траекторий и визуализация динамики системы.

## 2. Средства разработки

Для моделирования поставленной задачи используется язык Python со следующим набором библиотек:
"""

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as anim
from scipy.optimize import fsolve
from IPython.display import HTML

r"""
## 3. Математическая постановка

Система уравнений, описывающая движение трёх тел лишь за счёт их гравитационного взаимодействия, может быть записана следующим образом.

$$ 

\begin{cases}
    \dot{\bold{r_1}} = \bold{v_1},\\
    \dot{\bold{r_2}} = \bold{v_2},\\
    \dot{\bold{r_3}} = \bold{v_3},\\
    \dot{\bold{v_1}} = G\cdot\frac{m_2}{|r_{12}|^3}\cdot\overrightarrow{r_{12}} + G\cdot\frac{m_3}{|r_{13}|^3}\cdot\overrightarrow{r_{13}},\\
    \dot{\bold{v_2}} = G\cdot\frac{m_1}{|r_{21}|^3}\cdot\overrightarrow{r_{21}} + G\cdot\frac{m_3}{|r_{23}|^3}\cdot\overrightarrow{r_{23}},\\
    \dot{\bold{v_3}} = G\cdot\frac{m_1}{|r_{31}|^3}\cdot\overrightarrow{r_{31}} + G\cdot\frac{m_2}{|r_{32}|^3}\cdot\overrightarrow{r_{32}}.
\end{cases}

$$

, где $\bold{r_i}$ - радиус вектор $i$-ой материальной точки, а $\bold{v_i}$ - вектор скорости $i$-ой материальной точки. Правая часть данной системы описывается следующей функцией.
"""

def f(U, m1, m2, m3, G):
    ans = np.zeros(12)
    r12 = U[2:4] - U[ :2]
    r23 = U[4:6] - U[2:4]
    ans[ :2] = U[6:8]
    ans[2:4] = U[8:10]
    ans[4:6] = U[10:12]
    ans[6:8] = G*m2/np.power(np.linalg.norm(r12),3)*r12 + G*m3/np.power(np.linalg.norm(r12+r23),3)*(r12+r23)
    ans[8:10] = -G*m1/np.power(np.linalg.norm(r12),3)*r12 + G*m3/np.power(np.linalg.norm(r23),3)*r23
    ans[10:12] = -G*m2/np.power(np.linalg.norm(r23),3)*r23 - G*m1/np.power(np.linalg.norm(r12+r23),3)*(r12+r23)
    return ans

""" Конфигурация скоростей и положений материальных точек описывающих устойчивое положение "восьмерки" может быть представлено следующим образом. """

#PHYSICS
#обезразмеривающие параметры
tau = 1  # секунды в дне
q = 1    # масса земли (кг)
l = 1    # масштаб расстояния (м)

prm_vosem = {
    'r1' : np.array([0.970, -0.243], dtype=float),
    'r2' : np.array([-0.970, 0.243], dtype=float),
    'r3' : np.array([ 0.0, 0.0], dtype=float),
    'v1' : np.array([0.4662036850, 0.4323657300], dtype=float),
    'v2' : np.array([0.4662036850, 0.4323657300], dtype=float),
    'v3' : np.array([-0.93240737, -0.86473146], dtype=float),
    'm1' : 1,
    'm2' : 1,
    'm3' : 1,
    'G'  : 1
}

def pars_and_initiat(prm, t0, t1, dt, log=False):
    r1 = prm['r1'] / l
    r2 = prm['r2'] / l
    r3 = prm['r3'] / l
    v1 = prm['v1'] * tau / l
    v2 = prm['v2'] * tau / l
    v3 = prm['v3'] * tau / l
    m1 = prm['m1'] / q
    m2 = prm['m2'] / q
    m3 = prm['m3'] / q
    G  = prm['G'] * q * tau**2 / l**3

    if log:
        print(f"Масштабированные параметры:")
        print(f"r1 = {r1}, r2 = {r2}, r3 = {r3}")
        print(f"v1 = {v1}, v2 = {v2}, v3 = {v3}")
        print(f"m1 = {m1}, m2 = {m2}, m3 = {m3}")
        print(f"G = {G}")
    nsteps = math.ceil((t1 - t0) / dt)  # кол-во шагов
    if log:
        print(f"Количество шагов: {nsteps}")
    
    #PREPROCESSING
    r = np.zeros((nsteps+1,12))

    #INITIAL STATE
    r[0,:2] = r1
    r[0,2:4] = r2
    r[0,4:6] = r3
    r[0,6:8] = v1
    r[0,8:10] = v2
    r[0,10:12] = v3
    return r, nsteps, m1, m2, m3, G

r"""
## 4. Численные алгоритмы

Введем следующие обозначения:

$$
\bold{y} = \begin{pmatrix}
  \bold{r}_1 \\
  \bold{r}_2 \\
  \bold{r}_3 \\
  \bold{v}_1 \\
  \bold{v}_2 \\
  \bold{v}_3
\end{pmatrix}

, \quad \bold{f}(\bold{y}) = \begin{pmatrix}
    \bold{v_1}\\
    \bold{v_2}\\
    \bold{v_3}\\
    G\cdot\frac{m_2}{|r_{12}|^3}\cdot\overrightarrow{r_{12}} + G\cdot\frac{m_3}{|r_{13}|^3}\cdot\overrightarrow{r_{13}}\\
    G\cdot\frac{m_1}{|r_{21}|^3}\cdot\overrightarrow{r_{21}} + G\cdot\frac{m_3}{|r_{23}|^3}\cdot\overrightarrow{r_{23}}\\
    G\cdot\frac{m_1}{|r_{31}|^3}\cdot\overrightarrow{r_{31}} + G\cdot\frac{m_2}{|r_{32}|^3}\cdot\overrightarrow{r_{32}}
\end{pmatrix}
$$

Таким образом исходная система запишется:

$$
\dot{\bold{y}} = \bold{f}(\bold{y})
$$

, заметим, что правая часть явно не зависит от времени.

### Метод Рунге-Кутты 4-го порядка.

Методы вычисляет значение функиции на следующем шаге при помощи подсчёта промежуточных значений:

$$
\bold{y}_{n+1} = \bold{y}_n + \frac{dt}{6}(\bold{k}_1 + 2\bold{k}_2 + 2\bold{k}_3 + \bold{k}_4)\\
\quad\\
\bold{k}_1 = \bold{f}(t_n,\bold{y}_n)\\
\quad\\
\bold{k}_2 = \bold{f}(t_n + \frac{dt}{2}, \bold{y}_n + \frac{\bold{k}_1}{2})\\
\quad\\
\bold{k}_3 = \bold{f}(t_n + \frac{dt}{2}, \bold{y}_n + \frac{\bold{k}_2}{2})\\
\quad\\
\bold{k}_4 = \bold{f}(t_n + dt, \bold{y}_n + \bold{k}_3)
$$

Следом представлены функции реализующие шаг метода РК4 и решение системы ДУ методом РК4.
"""

def RK4(U, dt, m1, m2, m3, G):
    k1 = f(U, m1, m2, m3, G)
    k2 = f( U + k1*dt/2, m1, m2, m3, G)
    k3 = f( U + k2*dt/2, m1, m2, m3, G)
    k4 = f( U + k3*dt, m1, m2, m3, G)
    return U + (k1 + 2*k2 + 2*k3 + k4)*dt/6

def comp_RK4(prm, t0, t1, dt, log=False):
    r, nsteps, m1, m2, m3, G = pars_and_initiat(prm, t0, t1, dt, log)
    progress_step = nsteps // 10
    for frame in range(nsteps):
        r[frame+1] = RK4(r[frame], dt, m1, m2, m3, G)
        if log and ((frame + 1) % progress_step == 0 or frame == 0):
            progress = (frame + 1) / nsteps * 100
            print(f"Прогресс: {frame + 1}/{nsteps} ({progress:.1f}%)")
    return r, nsteps

r"""
### Метод Верлета.

Данный метод считает значения функции в целых точках, а значения скорости в полуцелых точках.

$$
\frac{\bold{y}_{n+1} - \bold{y}_n}{dt} = \bold{v}_{n+\frac{1}{2}} \\
\quad \\
\frac{ \bold{v}_{n+\frac{1}{2}} - \bold{v}_{n-\frac{1}{2}} }{dt} = \bold{a}_n
$$

Для старта данного метода необходимо задать скорость $\bold{v}_{\frac{1}{2}}$, которая может быть полученна следующим образом:

$$ \bold{v}_{\frac{1}{2}} = \bold{v}_{0} + \frac{dt}{2} \bold{a}_0 $$

Далее представлен участок кода реализующий шаг метода Верлета и решение системы ДУ методом Верлета.
"""

def verlet(U, V_half, dt, m1, m2, m3, G):
    U_next = U + dt*V_half
    f_next = f(U_next, m1, m2, m3, G)
    V_next_half = np.zeros(12)
    V_next_half[0:2] = V_half[0:2] + dt * f_next[6:8]
    V_next_half[2:4] = V_half[2:4] + dt * f_next[8:10]
    V_next_half[4:6] = V_half[4:6] + dt * f_next[10:12]
    return U_next, V_next_half

def comp_verlet(prm, t0, t1, dt, log=False):
    r, nsteps, m1, m2, m3, G = pars_and_initiat(prm, t0, t1, dt, log)
    V_half_hist = np.zeros_like(r)
    f_0 = f(r[0], m1, m2, m3, G)
    V_half_hist[0, :2] = r[0,6:8] + dt/2*f_0[6:8]
    V_half_hist[0,2:4] = r[0,8:10] + dt/2*f_0[8:10]
    V_half_hist[0,4:6] = r[0,10:12] + dt/2*f_0[10:12]
    progress_step = nsteps // 10
    for frame in range(nsteps):
        r[frame+1], V_half_hist[frame+1,:] = verlet(r[frame], V_half_hist[frame], dt, m1, m2, m3, G)
        r[frame+1,6:12] = (V_half_hist[frame+1,:6] + V_half_hist[frame,:6])/2
        if log and ((frame + 1) % progress_step == 0 or frame == 0):
            progress = (frame + 1) / nsteps * 100
            print(f"Прогресс: {frame + 1}/{nsteps} ({progress:.1f}%)")
    return r, nsteps

r"""
### Явный метод Адамса 4-го порядка

Шаг данного метода может быть записан следующим образом:

$$
\bold{y}_{n+1} = \bold{y}_{n} + \frac{dt}{24} \cdot (\quad 55 \cdot \bold{f}(\bold{y}_{n}) - 59 \cdot \bold{f}(\bold{y}_{n-1}) + 37 \cdot \bold{f}(\bold{y}_{n-2}) - 9 \cdot \bold{f}(\bold{y}_{n-3}) \quad )
$$

Для старта расчёта данным методом необходимо иметь значения функции на первых четырёх шагах по времени. В данной работе они получаются при помощи использования метода РК4.

Далее представлен участок кода реализующий шаг метода явного метода Адамча 4-го порядка и решение системы ДУ этим методом.
"""

def explicit_adams4(U, f_hist, dt, m1, m2, m3, G, frame):
    if frame < 3:
        y_next = RK4(U, dt, m1, m2, m3, G)
        return y_next, f(y_next, m1, m2, m3, G)
    y_next = U + dt/24*(55*f_hist[frame]-59*f_hist[frame-1]+37*f_hist[frame-2]-9*f_hist[frame-3])
    return y_next, f(y_next, m1, m2, m3, G)

def comp_explicit_adams4(prm, t0, t1, dt, log=False):
    r, nsteps, m1, m2, m3, G = pars_and_initiat(prm, t0, t1, dt, log)
    f_hist = np.zeros_like(r)
    f_hist[0,:] = f(r[0], m1, m2, m3, G)
    progress_step = nsteps // 10
    for frame in range(nsteps):
        r[frame+1], f_hist[frame+1,:] = explicit_adams4(r[frame], f_hist, dt, m1, m2, m3, G, frame)
        if log and ((frame + 1) % progress_step == 0 or frame == 0):
            progress = (frame + 1) / nsteps * 100
            print(f"Прогресс: {frame + 1}/{nsteps} ({progress:.1f}%)")
    return r, nsteps

r"""
### Неявный метод Адамса 4-го порядка

Значения функции на следующем шаге по времени получается в результате итерационного рпоцесса:

1) Начальное приближение:

$$
\bold{y}_{n+1}^{(0)} = \bold{y}_{n} + \frac{dt}{24} \cdot (\quad 55 \cdot \bold{f}(\bold{y}_{n}) - 59 \cdot \bold{f}(\bold{y}_{n-1}) + 37 \cdot \bold{f}(\bold{y}_{n-2}) - 9 \cdot \bold{f}(\bold{y}_{n-3}) \quad )
$$

2) Шаг итерации:

$$
\bold{y}_{n+1}^{(k+1)} = \bold{y}_{n} + \frac{dt}{24} \cdot (\quad 9 \cdot \bold{f}(\bold{y}_{n+1}^{(k)}) + 19 \cdot \bold{f}(\bold{y}_{n}) - 5 \cdot \bold{f}(\bold{y}_{n-1}) + \bold{f}(\bold{y}_{n-2}) \quad )
$$

3) Критерий остановки:

$$
|| \bold{y}_{n+1}^{(k+1)} - \bold{y}_{n+1}^{(k)} || < \varepsilon
$$

4) В результате иетарционного процесса получаем:

$$
\bold{y}_{n+1} = \bold{y}_{n+1}^{(k+1)}
$$

Для старта расчёта данным методом необходимо иметь значения функции на первых трёх шагах по времени. В данной работе они получаются при помощи использования метода РК4.

Далее представлен участок кода реализующий шаг метода неявного метода Адамча 4-го порядка и решение системы ДУ этим методом.
"""

def implicit_adams4(U, f_hist, dt, m1, m2, m3, G, frame):
    if frame < 2:
        y_next = RK4(U, dt, m1, m2, m3, G)
        return y_next, f(y_next, m1, m2, m3, G)
    y_predict = explicit_adams4(U, f_hist, dt, m1, m2, m3, G, frame)[0]
    def implicit_eq(y_next):
            f_next = f(y_next, m1, m2, m3, G)
            return y_next - U - (dt / 24) * (
                9 * f_next +
                19 * f_hist[frame] -
                5 * f_hist[frame-1] +
                1 * f_hist[frame-2]
            )
    y_next = fsolve(implicit_eq, y_predict)
    return y_next, f(y_next, m1, m2, m3, G)

def comp_implicit_adams4(prm, t0, t1, dt, log=False):
    r, nsteps, m1, m2, m3, G = pars_and_initiat(prm, t0, t1, dt, log)
    f_hist = np.zeros_like(r)
    f_hist[0,:] = f(r[0], m1, m2, m3, G)
    progress_step = nsteps // 10
    for frame in range(nsteps):
        r[frame+1], f_hist[frame+1,:] = implicit_adams4(r[frame], f_hist, dt, m1, m2, m3, G, frame)
        if log and ((frame + 1) % progress_step == 0 or frame == 0):
            progress = (frame + 1) / nsteps * 100
            print(f"Прогресс: {frame + 1}/{nsteps} ({progress:.1f}%)")
    return r, nsteps

""" ## 5. Визуализация результатов расчёта """

def animate(r,fps=20,anim_time=10):
    step = 1 + math.floor(len(r)/(fps*anim_time))
    r = r[::step,]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    tr_runge = (
        ax.plot( r[:1,0] , r[:1,1] , color="green", linestyle="--", label='Траектория 1-го тела')[0],
        ax.plot( r[:1,2] , r[:1,3] , color="blue", linestyle="--", label='Траектория 2-го тела')[0],
        ax.plot( r[:1,4] , r[:1,5] , color="red", linestyle="--", label='Траектория 3-го тела')[0],
        ax.scatter(r[0,0],r[0,1], color="black", marker="o",label='Положение 1-го тела'),
        ax.scatter(r[0,2],r[0,3], color="black", marker="*",label='Положение 2-го тела'),
        ax.scatter(r[0,4],r[0,5], color="black", marker="p",label='Положение 3-го тела')
    )
    def update(frame):
        tr_runge[0].set_data(r[:frame+1,0], r[:frame+1,1])
        tr_runge[1].set_data(r[:frame+1,2], r[:frame+1,3])
        tr_runge[2].set_data(r[:frame+1,4], r[:frame+1,5])
        tr_runge[3].set_offsets(r[frame, :2])
        tr_runge[4].set_offsets(r[frame,2:4])
        tr_runge[5].set_offsets(r[frame,4:6])
        ax.relim()
        ax.autoscale_view()
        ax.axis('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Анимированные траектории тел', fontsize=14, fontweight='bold')
        ax.legend()
        return (tr_runge[0] , tr_runge[1] , tr_runge[2] , tr_runge[3], tr_runge[4], tr_runge[5])

    animt = anim.FuncAnimation(
        fig, 
        update, 
        frames=len(r),
        interval=1000/fps,  
        blit=True,
        repeat=True
    )
    
    plt.close(fig)
    return HTML(animt.to_jshtml())

""" ### Анимация движения планет """

p = prm_vosem
t0 = 0
t1 = 300
dt = 0.1
r, nsteps = comp_explicit_adams4(p,t0,t1,dt,True)
animate(r,fps=20,anim_time=50)

"""
## 6. Сравнение численных методов

Сравниваться численные методы будут на основании сохранения инвариантов системы: полной энергии, углового момента - на больших промежутках времени.
"""

p = prm_vosem
t0 = 0
t1 = 300
dt = 0.05
r_RK4, nsteps = comp_RK4(p,t0,t1,dt)
r_verlet, nsteps = comp_verlet(p,t0,t1,dt)
r_e_adams, nsteps = comp_explicit_adams4(p,t0,t1,dt)
r_i_adams, nsteps = comp_implicit_adams4(p,t0,t1,dt)
t_space = np.linspace(t0,t1,nsteps+1)

""" ### Энергия """

def comp_E(U, m1, m2, m3, G):
    r12 = U[:2] - U[2:4]
    r13 = U[:2] - U[4:6]
    r23 = U[2:4] - U[4:6]
    l12 = np.linalg.norm(r12)
    l13 = np.linalg.norm(r13)
    l23 = np.linalg.norm(r23)
    E1 = m1*(U[6]*U[6] + U[7]*U[7])/2 - G*m1*( m2/l12 + m3/l13 )
    E2 = m2*(U[8]*U[8] + U[9]*U[9])/2 - G*m2*( m1/l12 + m3/l23 )
    E3 = m3*(U[10]*U[10] + U[11]*U[11])/2 - G*m3*( m1/l13 + m2/l23 )
    E = E1 + E2 + E3
    return E, E1, E2, E3

E_RK4 = np.array([ np.array(comp_E(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_RK4[:, :12] ])
E_e_adams = np.array([ np.array(comp_E(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_e_adams[:, :12] ])
E_i_adams = np.array([ np.array(comp_E(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_i_adams[:, :12] ])
E_verlet = np.array([ np.array(comp_E(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_verlet[:, :12] ])

E_init = E_RK4[0,0]

""" #### Относительная ошибка полной энергии рассматриваемых методов """

error_rel_rk4 = np.abs((E_RK4[:,0]-E_init)/E_init)
error_rel_e_adams = np.abs((E_e_adams[:,0]-E_init)/E_init)
error_rel_i_adams = np.abs((E_i_adams[:,0]-E_init)/E_init)
error_rel_verlet = np.abs((E_verlet[:,0]-E_init)/E_init)

fig, axs = plt.subplots(figsize=(10, 6))
axs.plot(t_space,error_rel_rk4,'r--',label='RK4')
axs.plot(t_space,error_rel_verlet,'g--',label='Verlet')
axs.plot(t_space,error_rel_e_adams,'b--',label='Explicit Adams 4')
axs.plot(t_space,error_rel_i_adams,'k--',label='Implicit Adams 4')

axs.set_title('Относительная ошибка методов с течением времени')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.legend()
axs.grid()
plt.show(fig)

fig, axs = plt.subplots(figsize=(10, 6))
graf_start = 0
graft_end = 51
axs.plot(t_space[graf_start:graft_end],error_rel_rk4[graf_start:graft_end],'r--',label='RK4')
axs.plot(t_space[graf_start:graft_end],error_rel_verlet[graf_start:graft_end],'g--',label='Verlet')
axs.plot(t_space[graf_start:graft_end],error_rel_e_adams[graf_start:graft_end],'b-.',label='Explicit Adams 4')
axs.plot(t_space[graf_start:graft_end],error_rel_i_adams[graf_start:graft_end],'k-.',label='Implicit Adams 4')

axs.set_title(f'Относительная ошибка методов за первые {graft_end-graf_start-1} временных единиц')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.legend()
axs.grid()
plt.show(fig)

fig, axs = plt.subplots(figsize=(10, 6))
graf_start = -51
graft_end = -1
axs.plot(t_space[graf_start:graft_end],error_rel_rk4[graf_start:graft_end],'r--',label='RK4')
axs.plot(t_space[graf_start:graft_end],error_rel_verlet[graf_start:graft_end],'g--',label='Verlet')
axs.plot(t_space[graf_start:graft_end],error_rel_e_adams[graf_start:graft_end],'b-.',label='Explicit Adams 4')
axs.plot(t_space[graf_start:graft_end],error_rel_i_adams[graf_start:graft_end],'k-.',label='Implicit Adams 4')

axs.set_title(f'Относительная ошибка методов за последние {graft_end-graf_start-1} временных единиц')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.legend()
axs.grid()
plt.show(fig)

""" #### Абсолютная ошибка полной энергии """

error_abs_rk4 = np.abs((E_RK4[:,0]-E_init))
error_abs_e_adams = np.abs((E_e_adams[:,0]-E_init))
error_abs_i_adams = np.abs((E_i_adams[:,0]-E_init))
error_abs_verlet = np.abs((E_verlet[:,0]-E_init))

fig, axs = plt.subplots(figsize=(10, 6))
axs.plot(t_space,error_abs_rk4,'r--',label='RK4')
axs.plot(t_space,error_abs_verlet,'g--',label='Verlet')
axs.plot(t_space,error_abs_e_adams,'b--',label='Explicit Adams 4')
axs.plot(t_space,error_abs_i_adams,'k--',label='Implicit Adams 4')

axs.set_title('Абсолютная ошибка методов с течением времени')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.legend()
axs.grid()
plt.show(fig)

fig, axs = plt.subplots(figsize=(10, 6))
graf_start = 0
graft_end = 101
axs.plot(t_space[graf_start:graft_end],error_abs_rk4[graf_start:graft_end],'r--',label='RK4')
axs.plot(t_space[graf_start:graft_end],error_abs_verlet[graf_start:graft_end],'g--',label='Verlet')
axs.plot(t_space[graf_start:graft_end],error_abs_e_adams[graf_start:graft_end],'b-.',label='Explicit Adams 4')
axs.plot(t_space[graf_start:graft_end],error_abs_i_adams[graf_start:graft_end],'k-.',label='Implicit Adams 4')

axs.set_title(f'Абсолютная ошибка методов за первые {graft_end-graf_start-1} временных единиц')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.legend()
axs.grid()
plt.show(fig)

fig, axs = plt.subplots(figsize=(10, 6))
graf_start = -51
graft_end = -1
axs.plot(t_space[graf_start:graft_end],error_abs_rk4[graf_start:graft_end],'r--',label='RK4')
axs.plot(t_space[graf_start:graft_end],error_abs_verlet[graf_start:graft_end],'g--',label='Verlet')
axs.plot(t_space[graf_start:graft_end],error_abs_e_adams[graf_start:graft_end],'b-.',label='Explicit Adams 4')
axs.plot(t_space[graf_start:graft_end],error_abs_i_adams[graf_start:graft_end],'k-.',label='Implicit Adams 4')

axs.set_title(f'Абсолютная ошибка методов за последние {graft_end-graf_start-1} временных единиц')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.legend()
axs.grid()
plt.show(fig)

""" ### Угловой момент """

def comp_L(U, m1, m2, m3, G):
    L1 = m1*(U[0]*U[7] - U[1]*U[6])
    L2 = m2*(U[2]*U[9] - U[3]*U[8])
    L3 = m3*(U[4]*U[11] - U[5]*U[10])
    L = L1 + L2 + L3
    return L, L1, L2, L3

L_RK4 = np.array([ np.array(comp_L(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_RK4[:, :12] ])
L_e_adams = np.array([ np.array(comp_L(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_e_adams[:, :12] ])
L_i_adams = np.array([ np.array(comp_L(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_i_adams[:, :12] ])
L_verlet = np.array([ np.array(comp_L(v,p['m1'],p['m2'],p['m3'],p['G'])) for v in r_verlet[:, :12] ])

L_init = L_RK4[0,0]
print(L_init)

"""
#### Абсолютная ошибка углового момента

Поскольку в конфигурации "восьмерки" угловой момент равен нулю, то рассматривается только абсолютная ошибка.
"""

error_abs_ang_rk4 = np.abs((L_RK4[:,0]-L_init))
error_abs_ang_e_adams = np.abs((L_e_adams[:,0]-L_init))
error_abs_ang_i_adams = np.abs((L_i_adams[:,0]-L_init))
error_abs_ang_verlet = np.abs((L_verlet[:,0]-L_init))

fig, axs = plt.subplots(figsize=(10, 6))
axs.plot(t_space,error_abs_ang_rk4,'r--',label='RK4')
axs.plot(t_space,error_abs_ang_verlet,'g--',label='Verlet')
axs.plot(t_space,error_abs_ang_e_adams,'b--',label='Explicit Adams 4')
axs.plot(t_space,error_abs_ang_i_adams,'k--',label='Implicit Adams 4')

axs.set_title('Абсолютная ошибка методов с течением времени')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.legend()
axs.grid()
plt.show(fig)

fig, axs = plt.subplots(figsize=(10, 6))
graf_start = 0
graft_end = 101
axs.plot(t_space[graf_start:graft_end],error_abs_ang_rk4[graf_start:graft_end],'r--',label='RK4')
axs.plot(t_space[graf_start:graft_end],error_abs_ang_verlet[graf_start:graft_end],'g--',label='Verlet')
axs.plot(t_space[graf_start:graft_end],error_abs_ang_e_adams[graf_start:graft_end],'b-.',label='Explicit Adams 4')
axs.plot(t_space[graf_start:graft_end],error_abs_ang_i_adams[graf_start:graft_end],'k-.',label='Implicit Adams 4')

axs.set_title(f'Абсолютная ошибка методов за первые {graft_end-graf_start-1} временных единиц')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.legend()
axs.grid()
plt.show(fig)

fig, axs = plt.subplots(figsize=(10, 6))
graf_start = -51
graft_end = -1
axs.plot(t_space[graf_start:graft_end],error_abs_ang_rk4[graf_start:graft_end],'r--',label='RK4')
axs.plot(t_space[graf_start:graft_end],error_abs_ang_verlet[graf_start:graft_end],'g--',label='Verlet')
axs.plot(t_space[graf_start:graft_end],error_abs_ang_e_adams[graf_start:graft_end],'b-.',label='Explicit Adams 4')
axs.plot(t_space[graf_start:graft_end],error_abs_ang_i_adams[graf_start:graft_end],'k-.',label='Implicit Adams 4')

axs.set_title(f'Абсолютная ошибка методов за последние {graft_end-graf_start-1} временных единиц')
axs.set_xlabel('t')
axs.set_ylabel('E')
axs.legend()
axs.grid()
plt.show(fig)

r"""
## 7. Анализ хаотичности системы при помощи параметра Ляпунова и его визуализация

### Показатель Ляпунова

Показатель Ляпунова [7] расчитывается следующим образом.

1) Инициализация
   $$ X_0 = X(0) $$
   $$ \tilde{X_0} = X(0) + \delta_0 \cdot e, \quad ||e|| = 1$$
2) Интегрирование
   $$ \dot{X} = f(X) $$
   $$ \dot{\tilde{X}} = f(\tilde{X}) $$
3) Фиксация возмущения на $k$-ом шаге
   $$ \delta X_k = \tilde{X}_k - X_k $$
4) Перенормировка
   $$ \tilde{X}_k = X_k + \delta_0 \frac{\delta X_k}{||\delta X_k||} $$
5) Вычисление параметра Ляпунова
   $$\lambda = \frac{1}{T}\sum_{k=0}^{N} \ln\frac{\delta X_k}{\delta_0}$$
"""

def pok_lyap_RK4(p, t0, t1, dt, delta=1.e-1, log=False): 
    delta0 = np.random.random(12)
    delta0 = delta*delta0 / np.linalg.norm(delta0)
    
    r1, nsteps, m1, m2, m3, G = pars_and_initiat(p, t0, t1, dt, log)
    r2 = np.copy(r1)
    r2[0] += delta0

    deltas = np.zeros(nsteps+1)
    deltas[0] = delta
    
    progress_step = nsteps // 10
    for frame in range(nsteps):
        r1[frame+1] = RK4(r1[frame], dt, m1, m2, m3, G)
        r2[frame+1] = RK4(r2[frame], dt, m1, m2, m3, G)
        deltas[frame+1] = np.linalg.norm(r2[frame+1] - r1[frame+1])
        r2[frame+1] = r1[frame+1] + delta*(r2[frame+1] - r1[frame+1])/deltas[frame+1]
        
        
        if log and ((frame + 1) % progress_step == 0 or frame == 0):
            progress = (frame + 1) / nsteps * 100
            print(f"Прогресс: {frame + 1}/{nsteps} ({progress:.1f}%)")
    lyap = 0
    for i in range(len(deltas)):
        lyap += np.log(deltas[i]/delta)
    lyap /= t1-t0
    return lyap, r1, r2, deltas, nsteps

""" ### Вычисление показателя Ляпунова на методе РК4 и его визуализация """

p = prm_vosem
t0 = 0
t1 = 300
dt = 0.05
delta = 1.e-3
lyap, r1, r2, deltas, nsteps = pok_lyap_RK4(p,t0,t1,dt, delta)
print(f'Показатель Ляпунова: {lyap}')

fig, axs = plt.subplots(figsize=(10, 4))

axs.plot( r1[:,0] , r1[:,1] , "r-" , label='Траектория 1-го тела')
axs.plot( r2[:,0] , r2[:,1] , "b-.", label='Траектория 1-го тела с возмущением', alpha=0.5)
axs.legend()
axs.set_title('Устойчивость движения 1-го тела', fontsize=14, fontweight='bold')
axs.set_xlabel('X')
axs.set_ylabel('Y')
plt.tight_layout()
plt.show()

r"""
## 8. Обсуждение результатов.

Сравнение численных методов производилось путем оценки их способности сохранять инварианты системы: полную энергию и угловой момент. РК4, Верлет, неявный Адамс сохранили указанные постоянные системы. При явном методе Адамса планеты начали сближаться со временем, что является свидетельством роста энергии в системе. Также стоит заметить, что метод Верлета, как и остальные методы, имеет периодический характер, однако со временем его период начинает смещаться, что можно заметить по несовпадению пиков в позднии моменты времени. Данные методы сохраняют угловой момент также хорошо как и энергию, кроме явного Адамса. Расчитанный показатель Ляпунова оказался равным $0.02 << 1$, что является достаточным доказательством устойчивости рассматриваемой конфигурации планет.

## 9. Заключение.

В ходе исследования было проведено сравнение нескольких численных методов (РК4, Верлет, явный и неявный методы Адамса) с точки зрения их способности сохранять ключевые инварианты динамической системы — полную энергию и угловой момент.

Основные выводы:

- Методы РК4, Верлет и неявный Адамс продемонстрировали хорошую способность сохранять как полную энергию, так и угловой момент системы — их можно считать пригодными для моделирования орбитальных движений.
- Явный метод Адамса показал неудовлетворительные результаты: со временем наблюдалось сближение планет, что свидетельствует о росте энергии в системе и нарушении её инвариантов.
- Метод Верлета, несмотря на в целом удовлетворительное сохранение инвариантов, имеет особенность — со временем происходит смещение периода колебаний (что заметно по несовпадению пиков в поздние моменты времени).

Перспективы развития работы:
- расширение модели до задачи N тел с целью проверки устойчивости методов в условиях взаимодействия большего числа небесных объектов;
- учёт дополнительных физических факторов (реальная форма небесных тел, релятивистские эффекты) для повышения реалистичности моделирования.

## Список использованной литературы.

[1] Арнольд В. И., Козлов В. В., Нейштадт А. И. *Математические аспекты классической и небесной механики*. — М.: УРСС, 2009.

[2] Ландау Л. Д., Лифшиц Е. М. *Механика*. Том 1 курса теоретической физики. — М.: Физматлит, 2004.

[3] Бахвалов Н. С., Жидков Н. П., Кобельков Г. М. *Численные методы*. — М.: Лаборатория знаний, 2020.

[4] Самарский А. А., Гулин А. В. *Численные методы*. — М.: Наука, 1989.

[5] Шмидт В. В. *Введение в компьютерное моделирование физических процессов*. — М.: МФТИ, 2010.

[6] Verlet L. Computer "Experiments" on Classical Fluids // Physical Review. 1967 \

[7] Шустер Г. *Детерминированный хаос. Введение*. — М.: Мир, 1988.
"""