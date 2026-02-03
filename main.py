import numpy as np
import matplotlib.pyplot as plt
from math import log, exp, sqrt, erf, pi

# 1. Black–Scholes benchmark

def norm_cdf(x):
    """Standard normal CDF (scalar)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_call_price(S0, K, r, sigma, T):
    """Black–Scholes price of a European call."""
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)

def simulate_gbm_terminal(S0, r, sigma, T, Z):
    """Terminal GBM price S_T given standard normal Z (can be array)."""
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

# 2. Monte Carlo estimators

def mc_crude_call(S0, K, r, sigma, T, N, rng):
    Z = rng.standard_normal(N)
    S_T = simulate_gbm_terminal(S0, r, sigma, T, Z)
    payoff = np.maximum(S_T - K, 0.0)
    return np.exp(-r * T) * payoff.mean()

def mc_antithetic_call(S0, K, r, sigma, T, N, rng):
    N2 = N // 2
    Z = rng.standard_normal(N2)
    S_plus  = simulate_gbm_terminal(S0, r, sigma, T, Z)
    S_minus = simulate_gbm_terminal(S0, r, sigma, T, -Z)
    payoff_pair = 0.5 * (np.maximum(S_plus - K, 0.0) +
                         np.maximum(S_minus - K, 0.0))
    return np.exp(-r * T) * payoff_pair.mean()

def mc_control_variate_call(S0, K, r, sigma, T, N, rng):
    Z = rng.standard_normal(N)
    S_T = simulate_gbm_terminal(S0, r, sigma, T, Z)
    Y = np.exp(-r * T) * np.maximum(S_T - K, 0.0)
    X = np.exp(-r * T) * S_T
    mu_X = S0
    cov_YX = np.cov(Y, X, ddof=1)[0, 1]
    var_X  = np.var(X, ddof=1)
    beta_hat = cov_YX / var_X

    Y_star = Y - beta_hat * (X - mu_X)
    return Y_star.mean()

def mc_importance_call(S0, K, r, sigma, T, N, mu_IS, rng):
    Z_shifted = rng.normal(loc=mu_IS, scale=1.0, size=N)
    S_T = simulate_gbm_terminal(S0, r, sigma, T, Z_shifted)
    payoff = np.maximum(S_T - K, 0.0)
    weights = np.exp(-mu_IS * Z_shifted + 0.5 * mu_IS**2)
    discounted = np.exp(-r * T) * payoff * weights
    return discounted.mean()

# 3. Experiment setup

# Model parameters
S0    = 100.0
K     = 110.0   # slightly out-of-the-money so IS has something to do
r     = 0.05
sigma = 0.2
T     = 1.0

mu_IS = 2.0     # importance-sampling shift

true_price = bs_call_price(S0, K, r, sigma, T)
print("Black–Scholes reference price:", true_price)

# Sample sizes to test
Ns = np.array([10**2, 5*10**2, 10**3, 5*10**3, 10**4, 5*10**4, 10**5])

M = 50  # number of independent runs for RMSE/VRF

nN = len(Ns)
prices_crude = np.zeros((nN, M))
prices_anti  = np.zeros((nN, M))
prices_cv    = np.zeros((nN, M))
prices_is    = np.zeros((nN, M))

rng_master = np.random.default_rng(2024)

for i, N in enumerate(Ns):
    for m in range(M):
        rng = np.random.default_rng(rng_master.integers(0, 2**32-1))
        prices_crude[i, m] = mc_crude_call(S0, K, r, sigma, T, N, rng)
        prices_anti[i, m]  = mc_antithetic_call(S0, K, r, sigma, T, N, rng)
        prices_cv[i, m]    = mc_control_variate_call(S0, K, r, sigma, T, N, rng)
        prices_is[i, m]    = mc_importance_call(S0, K, r, sigma, T, N, mu_IS, rng)

# 4. Compute RMSE, VRF, and bias

def rmse(estimates, truth):
    return np.sqrt(np.mean((estimates - truth)**2, axis=1))

rmse_crude = rmse(prices_crude, true_price)
rmse_anti  = rmse(prices_anti,  true_price)
rmse_cv    = rmse(prices_cv,    true_price)
rmse_is    = rmse(prices_is,    true_price)

var_crude = np.var(prices_crude, axis=1, ddof=1)
var_anti  = np.var(prices_anti,  axis=1, ddof=1)
var_cv    = np.var(prices_cv,    axis=1, ddof=1)
var_is    = np.var(prices_is,    axis=1, ddof=1)

vrf_anti = var_crude / var_anti
vrf_cv   = var_crude / var_cv
vrf_is   = var_crude / var_is

bias_crude = np.mean(prices_crude - true_price, axis=1)
bias_anti  = np.mean(prices_anti  - true_price, axis=1)
bias_cv    = np.mean(prices_cv    - true_price, axis=1)
bias_is    = np.mean(prices_is    - true_price, axis=1)

# 5. Figure 1 – RMSE vs N (log–log)

plt.figure(figsize=(7, 5))
plt.loglog(Ns, rmse_crude, 'o-', label='Crude MC')
plt.loglog(Ns, rmse_anti,  's-', label='Antithetic')
plt.loglog(Ns, rmse_cv,    '^-', label='Control Variates')
plt.loglog(Ns, rmse_is,    'd-', label='Importance Sampling')

theory_line = rmse_crude[0] * np.sqrt(Ns[0]) / np.sqrt(Ns)
plt.loglog(Ns, theory_line, 'k--', label=r'Theoretical $1/\sqrt{N}$')

plt.xlabel('Number of Samples $N$')
plt.ylabel('RMSE')
plt.title('Figure 1: RMSE vs $N$ (log–log)')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.show()

# 6. Figure 2 – Variance Reduction Factor vs N

plt.figure(figsize=(7, 5))
plt.semilogx(Ns, vrf_anti, 's-', label='Antithetic')
plt.semilogx(Ns, vrf_cv,   '^-', label='Control Variates')
plt.semilogx(Ns, vrf_is,   'd-', label='Importance Sampling')

plt.axhline(1.0, color='k', linestyle='--', linewidth=1)
plt.xlabel('Number of Samples $N$')
plt.ylabel('Variance Reduction Factor (VRF)')
plt.title('Figure 2: Variance Reduction Factor vs $N$')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.show()

print("Approximate mean VRFs over all N:")
print("  Antithetic        ~ {:.2f}".format(vrf_anti.mean()))
print("  Control variates  ~ {:.2f}".format(vrf_cv.mean()))
print("  Importance samp.  ~ {:.2f}".format(vrf_is.mean()))

# 7. Figure 3 – Estimator stability (boxplots at largest N)

idx = -1  # index for Ns[-1]
N_star = Ns[idx]

data_box = [
    prices_crude[idx, :],
    prices_anti[idx, :],
    prices_cv[idx, :],
    prices_is[idx, :]
]

labels = ['Crude MC', 'Antithetic', 'Control Var.', 'Imp. Samp.']

plt.figure(figsize=(7, 5))
plt.boxplot(data_box, labels=labels, showmeans=True)
plt.axhline(true_price, color='k', linestyle='--', linewidth=1,
            label='True price')

plt.ylabel('Estimated price')
plt.title(f'Figure 3: Estimator Distribution at N = {N_star}')
plt.grid(axis='y', linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()
