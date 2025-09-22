import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as anim
def unmer(r1, r2, v1, v2, m1, m2, G, l, tau, q):
    r1 = r1 / l
    r2 = r2 / l
    v1 = v1 * tau / l
    v2 = v2 * tau / l
    m1 = m1 / q
    m2 = m2 / q
    G = G * q * tau**2 / l**3
    return r1, r2, v1, v2, m1, m2, G
def start(r,r1,r2,v1,v2):
    r[0, :3] = r1
    r[0,3:6] = r2
    r[0,6:9] = v1
    r[0,9:12] = v2
def inic_lines(ax,r):
    arr = (
        ax.plot   (r[:1,0],r[:1,1],color="green", linestyle="--", label='Траектория 1-го тела')[0],
        ax.plot   (r[:1,2],r[:1,3],color="orange", linestyle="--", label='Траектория 2-го тела')[0],
        ax.scatter(r[:1,0],r[:1,1],color="green", marker="."),
        ax.scatter(r[:1,2],r[:1,3],color="orange", marker="."),
        ax.scatter(r[:1,0],r[:1,1],color="black", marker="o",label='Положение 1-го тела'),
        ax.scatter(r[:1,2],r[:1,3],color="black", marker="*",label='Положение 2-го тела')
    )
    return arr
def orthonormal_basis_from_normal(n):
    n = np.asarray(n, dtype=float)
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("Нулевая нормаль")
    nhat = n / norm
    ax = np.abs(nhat)
    if ax[0] <= ax[1] and ax[0] <= ax[2]:
        a = np.array([1.0, 0.0, 0.0])
    elif ax[1] <= ax[0] and ax[1] <= ax[2]:
        a = np.array([0.0, 1.0, 0.0])
    else:
        a = np.array([0.0, 0.0, 1.0])
    u = a - np.dot(a, nhat) * nhat
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        raise RuntimeError("Вспомогательный вектор вырожден; попробуйте другой a")
    uhat = u / u_norm
    vhat = np.cross(nhat, uhat)
    return uhat, vhat, nhat
