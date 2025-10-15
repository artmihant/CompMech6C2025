import numpy as np
import matplotlib.pyplot as plt

g = 9.81
k = 0.01
wx = 2.0
v0 = 50.0
x_target = 200.0
y_target = 0.0
dt = 0.01
t_max = 20.0
tol = 1e-6

def f(s):
    x,y,vx,vy = s
    return np.array([vx, vy, -k*(vx-wx), -g - k*vy])

def rk4(s):
    k1 = dt*f(s)
    k2 = dt*f(s+0.5*k1)
    k3 = dt*f(s+0.5*k2)
    k4 = dt*f(s+k3)
    return s + (k1+2*k2+2*k3+k4)/6

def integrate(theta):
    vx0 = v0*np.cos(theta)
    vy0 = v0*np.sin(theta)
    s = np.array([0.0, 0.0, vx0, vy0])
    traj = [s.copy()]
    t = 0.0
    while t < t_max:
        s_next = rk4(s)
        traj.append(s_next.copy())
        if (s[1]-y_target)*(s_next[1]-y_target) <= 0 and t>0:
            a = (y_target - s[1])/(s_next[1]-s[1] + 1e-16)
            s_cross = s + a*(s_next - s)
            s_cross[1] = y_target
            traj[-1] = s_cross
            s = s_cross
            break
        s = s_next
        t += dt
    return np.array(traj)

def residual(theta):
    traj = integrate(theta)
    return traj[-1,0] - x_target

def bisection(l, r, max_it=50):
    tl, tr = l, r
    rl, rr = residual(tl), residual(tr)
    hist, res = [tl], [rl]
    for _ in range(max_it):
        if rl*rr > 0: break
        tm = 0.5*(tl+tr)
        rm = residual(tm)
        hist.append(tm); res.append(rm)
        if rl*rm <= 0: tr, rr = tm, rm
        else: tl, rl = tm, rm
        if abs(rm) < tol: break
    return tm, hist, res

def regula_falsi(l, r, max_it=50):
    tl, tr = l, r
    rl, rr = residual(tl), residual(tr)
    hist, res = [tl], [rl]
    for _ in range(max_it):
        tm = (rr*tl - rl*tr)/(rr-rl)
        rm = residual(tm)
        hist.append(tm); res.append(rm)
        if rl*rm <= 0: tr, rr = tm, rm
        else: tl, rl = tm, rm
        if abs(rm) < tol: break
    return tm, hist, res

def newton(t0, max_it=50, h=1e-6):
    t = t0
    hist, res = [t], [residual(t)]
    for _ in range(max_it):
        r = residual(t)
        dr = (residual(t+h)-residual(t-h))/(2*h)
        if abs(dr) < 1e-12: break
        t -= r/dr
        hist.append(t); res.append(residual(t))
        if abs(res[-1]) < tol: break
    return t, hist, res

l, r = np.radians(5), np.radians(60)
t0 = np.radians(35)
tb, hb, rb = bisection(l, r)
trf, hrf, rrf = regula_falsi(l, r)
tn, hn, rn = newton(t0)

plt.figure(figsize=(10,5))
plt.plot(np.abs(rb), label='Дихотомия')
plt.plot(np.abs(rrf), label='Regula Falsi')
plt.plot(np.abs(rn), label='Метод Ньютона')
plt.yscale('log')
plt.xlabel('Итерация')
plt.ylabel('|residual|')
plt.title('Сходимость методов')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
for th,name in [(tb,'Дихотомия'),(trf,'Regula Falsi'),(tn,'Ньютон')]:
    traj = integrate(th)
    plt.plot(traj[:,0], traj[:,1], label=name)
plt.scatter([0],[0],s=60)
plt.scatter([x_target],[y_target],s=80,marker='x',color='red')
plt.xlabel('x, м')
plt.ylabel('y, м')
plt.title('Траектории при найденных углах')
plt.legend()
plt.grid(True,alpha=0.3)
plt.axis('equal')
plt.show()
