import numpy as np

prm_Earth_Sun = {
    'r1' : np.array([-1.5e9, 0, 0], dtype=float),
    'r2' : np.array([ 0, 0, 0], dtype=float),
    'v1' : np.array([0, -5e4, 0], dtype=float),
    'v2' : np.array([0, 0, 1e5], dtype=float),
    'm1' : 5.97e24,
    'm2' : 5.97e29
}
prm_Earth_Jupiter = {
    'r1' : np.array([-4.25e8, 0, 0], dtype=float),
    'r2' : np.array([ 4.25e8, 0, 0], dtype=float),
    'v1' : np.array([0, 12e3, 0], dtype=float),
    'v2' : np.array([0, -38, 0], dtype=float),
    'm1' : 5.97e24,
    'm2' : 1.8982e27
}
prm_Earth_Mars = {
    'r1' : np.array([-39000000.0, 0, 0], dtype=float),
    'r2' : np.array([ 39000000.0, 0, 0], dtype=float),
    'v1' : np.array([0.0, 6.52e3, 0.0], dtype=float),
    'v2' : np.array([0.0, -7.38e3, 0.0], dtype=float),
    'm1' : 5.97e24,
    'm2' : 6.4171e23
}