import numpy as np
from scipy.stats import norm
def bs_call_price(S, K, tau, sigma, r):
    """
    Black-Scholes formula for European call.
    tau = time to maturity (T - t)
    S can be scalar or numpy array.
    """
    S = np.array(S, dtype=float)
    tau = np.maximum(tau, 1e-16)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

def thomas(a, b, c, d): # Метод прогонки
    """
    Thomas algorithm for tridiagonal system.
    a, b, c: arrays of length n (a[0] unused or zero, c[-1] unused or zero)
    d: RHS of length n
    Returns x of length n solving:
      a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i], i=0..n-1
    (We assume a[0] == 0 and c[-1] == 0)
    """
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / denom if i < n-1 else 0.0  # last c ignored
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

def solve_crank_nicolson(Smax, M, N_time, sigma, r, K, T): # Схема Кранка–Николсона
    """
    Crank-Nicolson solver. Returns S, tau, V (shape (N_time+1, M+1))
    """
    dx = Smax / M
    dt = T / N_time
    S = np.linspace(0, Smax, M+1)
    tau = np.linspace(0, T, N_time+1)
    V = np.zeros((N_time+1, M+1))
    V[0, :] = np.maximum(S - K, 0.0)
    for n in range(0, N_time):
        a = np.zeros(M-1); b = np.zeros(M-1); c = np.zeros(M-1); d = np.zeros(M-1)
        for j in range(M-1):
            i = j + 1
            Si = S[i]
            alpha = 0.5 * sigma**2 * Si**2
            beta = r * Si
            a[j] = -0.5 * dt * (alpha / dx**2 - beta / (2*dx))
            b[j] = 1.0 + 0.5 * dt * (2*alpha / dx**2 + r)
            c[j] = -0.5 * dt * (alpha / dx**2 + beta / (2*dx))
            # RHS coefficients (from previous time level)
            a_rhs = 0.5 * dt * (alpha / dx**2 - beta / (2*dx))
            b_rhs = 1.0 - 0.5 * dt * (2*alpha / dx**2 + r)
            c_rhs = 0.5 * dt * (alpha / dx**2 + beta / (2*dx))
            d[j] = a_rhs * V[n, i-1] + b_rhs * V[n, i] + c_rhs * V[n, i+1]
        # boundaries
        V0_next = 0.0
        VM_next = Smax - K * np.exp(-r * tau[n+1])
        # account for known boundary values in RHS
        d[0] -= a[0] * V0_next
        d[-1] -= c[-1] * VM_next
        V_inner = thomas(a, b, c, d)
        V[n+1, 0] = V0_next
        V[n+1, 1:M] = V_inner
        V[n+1, M] = VM_next
    return S, tau, V


def calculate_option_price(S0, K, T, r, sigma, Smax_ratio=3.0, M=2000, N_time=400):
    """
    Вычисляет справедливую цену европейского колл-опциона по схеме Кранка-Николсона
    """
    Smax = Smax_ratio * K
    
    # Решаем уравнение Кранка-Николсона
    S, tau, V = solve_crank_nicolson(Smax, M, N_time, sigma, r, K, T)
    
    # Находим цену при S0
    idx = np.argmin(np.abs(S - S0))
    option_price = V[-1, idx]
    
    return option_price

if __name__ == "__main__":

    S0 = 100.0      
    K = 100.0
    T = 1.0         
    r = 0.05        
    sigma = 0.2     
    
    fair_price = calculate_option_price(S0, K, T, r, sigma)
    print(f"Справедливая цена опциона: {fair_price:.4f}")
    
    analytic_price = bs_call_price(S0, K, T, sigma, r)
    print(f"Аналитическая цена: {analytic_price:.4f}")
    print(f"Разница: {abs(fair_price - analytic_price):.6f}")