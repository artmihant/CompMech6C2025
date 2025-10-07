import numpy as np
import time
import matplotlib.pyplot as plt

g = 9.81
k = 0.01
wind_x = 2.0
v0 = 50.0
x_target = 200.0
y_target = 0.0
x_start = 0.0
y_start = 0.0
t_max = 20.0
n_steps = 1200
dt = t_max / n_steps
tol = 1e-6
deg = np.degrees
rad = np.radians

def f(state):
    x,y,vx,vy = state
    return np.array([vx, vy, -k*(vx - wind_x), -g - k*vy])

def rk4_step(state):
    k1 = dt*f(state)
    k2 = dt*f(state+0.5*k1)
    k3 = dt*f(state+0.5*k2)
    k4 = dt*f(state+k3)
    return state + (k1+2*k2+2*k3+k4)/6

def init(theta):
    return np.array([x_start,y_start,v0*np.cos(theta),v0*np.sin(theta)])

def integrate(theta):
    s = init(theta).copy()
    tr = [s.copy()]
    t = 0.0
    while t < t_max and s[1] >= y_target:
        s_next = rk4_step(s)
        t += dt
        tr.append(s_next.copy())
        s = s_next
        if s[1] < y_target and tr[-2][1] >= y_target:
            a = tr[-2]
            b = tr[-1]
            r = (y_target - a[1])/(b[1]-a[1] + 1e-15)
            s = a + r*(b-a)
            s[1] = y_target
            tr[-1] = s.copy()
            break
    return np.array(tr)

fevals = {"rf":0,"newton":0}
def residual(theta, counter=None):
    tr = integrate(theta)
    if counter: fevals[counter]+=1
    return tr[-1,0] - x_target

def bracketing(a, b, max_iter=60):
    ra = residual(a,"rf")
    rb = residual(b,"rf")
    hist_t = [a, b]
    hist_r = [ra, rb]
    for _ in range(max_iter):
        if abs(ra) < tol: return a, hist_t, hist_r
        if abs(rb) < tol: return b, hist_t, hist_r
        if ra*rb > 0:
            c = (a+b)/2
            rc = residual(c,"rf")
            hist_t.append(c); hist_r.append(rc)
            if abs(rc) < tol: return c, hist_t, hist_r
            if abs(rc) > abs(ra) and abs(rc) > abs(rb):
                a -= rad(2); b += rad(2)
                ra = residual(a,"rf"); rb = residual(b,"rf")
                hist_t.extend([a,b]); hist_r.extend([ra,rb])
            else:
                if ra*rc < 0: b, rb = c, rc
                else: a, ra = c, rc
        else:
            break
    for _ in range(max_iter):
        c = (a*rb - b*ra)/(rb-ra)
        rc = residual(c,"rf")
        hist_t.append(c); hist_r.append(rc)
        if abs(rc) < tol: return c, hist_t, hist_r
        if ra*rc < 0: b, rb = c, rc
        else: a, ra = c, rc
        if abs(b-a) < 1e-12: break
    return (a if abs(ra)<abs(rb) else b), hist_t, hist_r

def newton(theta0, max_iter=30):
    t = theta0
    hist_t = [t]
    r = residual(t,"newton")
    hist_r = [r]
    for _ in range(max_iter):
        if abs(r) < tol: return t, hist_t, hist_r
        dr = (residual(t+1e-6,"newton") - residual(t-1e-6,"newton"))/(2e-6)
        if abs(dr) < 1e-14: break
        t = t - r/dr
        r = residual(t,"newton")
        hist_t.append(t); hist_r.append(r)
    return t, hist_t, hist_r

print("="*60)
print("SHOOTING METHODS")
print("="*60)
print(f"Target: x = {x_target:.1f}, y = {y_target:.1f}")
print(f"Initial speed: v0 = {v0:.1f}")
print(f"Air drag k = {k:.3f} , wind_x = {wind_x:.1f}")
print()

theta_l = rad(10.0)
theta_r = rad(60.0)
theta0  = rad(35.0)

fevals["rf"]=0
t0 = time.time()
theta_rf, th_rf, res_rf = bracketing(theta_l, theta_r, max_iter=80)
t1 = time.time()

fevals["newton"]=0
t2 = time.time()
theta_n, th_n, res_n = newton(theta0, max_iter=40)
t3 = time.time()

traj_rf = integrate(theta_rf)
traj_n  = integrate(theta_n)

print("-"*50)
print("Regula-Falsi/Bisection result")
print(f"Angle: {deg(theta_rf):.6f} deg")
print(f"Residual: {res_rf[-1]:.3e} m")
print(f"Iterations: {len(th_rf)}")
print(f"Function evaluations: {fevals['rf']}")
print(f"Wall time: {(t1-t0)*1000:.1f} ms")
print("-"*50)
print("Newton result")
print(f"Angle: {deg(theta_n):.6f} deg")
print(f"Residual: {res_n[-1]:.3e} m")
print(f"Iterations: {len(th_n)}")
print(f"Function evaluations: {fevals['newton']}")
print(f"Wall time: {(t3-t2)*1000:.1f} ms")
print("-"*50)
print("Comparison")
print(f"|theta_rf - theta_newton|: {abs(deg(theta_rf-theta_n)):.3e} deg")

def pack_trajs(thetas, limit=5):
    out = []
    m = min(limit, len(thetas))
    idx = np.linspace(0, len(thetas)-1, m, dtype=int)
    for i in idx:
        tr = integrate(thetas[i])
        out.append((thetas[i], tr))
    return out

rf_trajs = pack_trajs(th_rf, limit=5)
n_trajs  = pack_trajs(th_n,  limit=5)

plt.figure(figsize=(13,4))
plt.subplot(1,3,1)
for t,tr in rf_trajs:
    plt.plot(tr[:,0], tr[:,1], label=f"RF {deg(t):.1f}°")
for t,tr in n_trajs:
    plt.plot(tr[:,0], tr[:,1], linestyle="--", label=f"Newton {deg(t):.1f}°")
plt.scatter([x_start],[y_start], marker="o", s=30, label="Start")
plt.scatter([x_target],[y_target], marker="x", s=50, label="Target")
plt.xlabel("x, m")
plt.ylabel("y, m")
plt.title("Trajectories")
plt.legend(fontsize=8, loc="lower left")

plt.subplot(1,3,2)
plt.plot(range(len(th_rf)), deg(np.array(th_rf)), marker="o", label="RF")
plt.plot(range(len(th_n)), deg(np.array(th_n)), marker="o", label="Newton")
plt.xlabel("Iteration")
plt.ylabel("Angle, deg")
plt.title("Angle convergence")
plt.legend(fontsize=8)

plt.subplot(1,3,3)
plt.semilogy(range(len(res_rf)), np.abs(np.array(res_rf)), marker="o", label="RF |res|")
plt.semilogy(range(len(res_n)), np.abs(np.array(res_n)), marker="o", label="Newton |res|")
plt.xlabel("Iteration")
plt.ylabel("|Residual|, m")
plt.title("Residual convergence")
plt.legend(fontsize=8)

plt.tight_layout()
plt.show()
