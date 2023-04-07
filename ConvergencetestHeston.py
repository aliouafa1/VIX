import numpy as np
from scipy.stats import norm

# Heston model parameters
v0 = 0.04
theta = 0.04
kappa = 1.5
sigma = 0.3
rho = -0.5

# we Set VarSwap parameters
T = 1.0
K = 100.0

# we set Monte Carlo simulation parameters
N = 10000 # Number of paths (we can try with more paths)
M = 100 # Number of time steps
dt = T/M # Time step size
r = 0.0 # Risk-free rate

# we Define function to calculate VarSwap payoff
def varswap_payoff(paths, T, K):
    ST = paths[:, -1]
    return np.maximum((1/2)*(1/T)*np.sum((ST/K - 1)**2), 0)

# we Define function to calculate Monte Carlo estimate of VarSwap price
def varswap_monte_carlo(S0, v0, theta, kappa, sigma, rho, r, T, K, N, M):
    dt_sqrt = np.sqrt(dt)
    r_dt = r*dt
    v_dt_sqrt = sigma*np.sqrt(dt)
    lnS = np.log(S0)
    v = np.ones(N)*v0
    ST = np.zeros(N)
    for i in range(M):
        # Generate correlated Brownian motions
        dW1 = np.random.normal(size=N)*dt_sqrt
        dW2 = rho*dW1 + np.sqrt(1 - rho**2)*np.random.normal(size=N)*dt_sqrt
        # Update stock price and volatility
        lnS += (r - 0.5*v)*dt + v_dt_sqrt*dW1
        v += kappa*(theta - v)*dt + sigma*np.sqrt(v)*dW2
        # Store final stock prices
        if i == M-1:
            ST = np.exp(lnS)
    # Calculate VarSwap payoff
    payoff = varswap_payoff(np.vstack([np.ones(N)*S0, ST]).T, T, K)
    # Calculate Monte Carlo estimate of VarSwap price
    price = np.exp(-r*T)*payoff
    return price

# we Define function to calculate standard error of Monte Carlo estimate
def standard_error(data):
    n = len(data)
    return np.std(data, ddof=1)/np.sqrt(n)

# we Run convergence test
n_values = [1000, 2000, 4000, 8000, 16000, 32000]
prices = []
standard_errors = []
for n in n_values:
    price = varswap_monte_carlo(S0, v0, theta, kappa, sigma, rho, r, T, K, n, M)
    prices.append(price)
    standard_errors.append(standard_error(prices))
    print(f"n = {n}: VarSwap price = {price:.4f}, standard error = {standard_errors[-1]:.4f}")
