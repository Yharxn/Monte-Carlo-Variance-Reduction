import numpy as np
import matplotlib.pyplot as plt

def solve_explicit_pde(
    S0: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    M: int = 100, 
    N: int = 2000
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calculates European Call Option price using Explicit Finite Difference Method.
    
    Returns:
        tuple: (Estimated Price at S0, Stock Grid, Final Option Values)
    """
    # 1. Grid Setup
    S_max = 3.0 * K  # Standard practice: 3x Strike
    dS = S_max / M
    dt = T / N
    
    # Stability Check (CFL Condition)
    # If dt > dS^2 / (sigma^2 * S^2), the solution explodes.
    # We print a warning if this might happen at the strike price.
    cfl_val = dt / (dS**2)
    if cfl_val > 0.5: # Rough heuristic
        print(f"Warning: Time step might be too large (CFL check: {cfl_val:.2f})")

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

    # 2. Boundary Conditions
    # Payoff at maturity (t = T)
    V[N, :] = np.maximum(S_grid - K, 0)

    # Boundary at S = 0 (Stock worthless -> Option worthless)
    V[:, 0] = 0.0
    
    # Boundary at S = S_max (Deep in the money -> S - K*exp(-rt))
    time_points = np.linspace(0, T, N + 1)
    V[:, M] = S_max - K * np.exp(-r * (T - time_points))

    # 3. Time Stepping Loop (Backward from T to 0)
    # Pre-compute coefficients to optimize speed
    j = np.arange(1, M)  # Indices for internal nodes
    alpha = 0.5 * dt * (sigma**2 * j**2 - r * j)
    beta  = 1.0 - dt * (sigma**2 * j**2 + r)
    gamma = 0.5 * dt * (sigma**2 * j**2 + r * j)

    for n in range(N - 1, -1, -1):
        V[n, 1:M] = alpha * V[n + 1, 0:M - 1] + \
                    beta  * V[n + 1, 1:M]     + \
                    gamma * V[n + 1, 2:M + 1]

    # 4. Interpolate result
    price = np.interp(S0, S_grid, V[0, :])
    return price, S_grid, V[0, :]

def plot_results(S_grid: np.ndarray, V_curve: np.ndarray, K: float):
    """Generates the validation plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(S_grid, V_curve, label='PDE Solution (Finite Difference)', linewidth=2)
    plt.axvline(K, color='r', linestyle='--', alpha=0.6, label='Strike Price')
    plt.title("Deterministic PDE Validator vs Strike")
    plt.xlabel("Stock Price ($S$)")
    plt.ylabel("Option Value ($V$)")
    plt.xlim(K * 0.5, K * 1.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Simulation Parameters
    S0_input = 100.0
    K_input = 100.0
    T_input = 1.0
    r_input = 0.05
    sigma_input = 0.2
    
    print("Running PDE Validator...")
    price, grid, curve = solve_explicit_pde(S0_input, K_input, T_input, r_input, sigma_input)
    
    print(f"-" * 30)
    print(f"Validation Price: ${price:.5f}")
    print(f"-" * 30)
    
    plot_results(grid, curve, K_input)
