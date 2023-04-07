import numpy as np
from scipy.stats import norm

# We set up parameters
S0 = 100
K = 100
r = 0.05
q = 0.0
T = 1.0
v0 = 0.1**2
theta = 0.1**2
kappa = 2.0
sigma = 0.3
rho = -0.5

# We generate volatility surface
strikes = np.arange(50, 151, 10)
maturities = np.arange(0.5, 2.51, 0.5)
vol_surface = np.zeros((len(strikes), len(maturities)))
for i, K in enumerate(strikes):
    for j, T in enumerate(maturities):
        vol_surface[i, j] = heston_implied_vol(K, r, q, T, v0, theta, kappa, sigma, rho)

# w set up VarSwap and VolSwap parameters
timestep = 1000
num_paths = 100000
varswap_maturity = 2.0
varswap_notional = 1.0
volswap_maturity = 1.0
volswap_notional = 1.0

# we calculate VarSwap and VolSwap fair values using Monte Carlo with Antithetic Variates
varswap_mc = np.zeros(len(strikes))
volswap_mc = np.zeros(len(strikes))
for i, K in enumerate(strikes):
    varswap_mc[i] = monte_carlo_varswap_price(K, r, q, varswap_maturity, timestep, num_paths, S0, v0, theta, kappa, sigma, rho, antithetic=True)
    volswap_mc[i] = monte_carlo_volswap_price(K, r, q, volswap_maturity, timestep, num_paths, S0, v0, theta, kappa, sigma, rho, antithetic=True)

# we Calculate VarSwap and VolSwap fair values using Monte Carlo with Control Variates
varswap_mc_cv = np.zeros(len(strikes))
volswap_mc_cv = np.zeros(len(strikes))
varswap_cv_beta = np.zeros(len(strikes))
volswap_cv_beta = np.zeros(len(strikes))
for i, K in enumerate(strikes):
    varswap_mc_cv[i], varswap_cv_beta[i] = monte_carlo_varswap_price_cv(K, r, q, varswap_maturity, timestep, num_paths, S0, v0, theta, kappa, sigma, rho, vol_surface, antithetic=True)
    volswap_mc_cv[i], volswap_cv_beta[i] = monte_carlo_volswap_price_cv(K, r, q, volswap_maturity, timestep, num_paths, S0, v0, theta, kappa, sigma, rho, vol_surface, antithetic=True)

# we Calculate VarSwap and VolSwap fair values using numerical integration
varswap_numint = np.zeros(len(strikes))
volswap_numint = np.zeros(len(strikes))
for i, K in enumerate(strikes):
    varswap_numint[i] = numerical_varswap_price(K, r, q, varswap_maturity, timestep, S0, v0, theta, kappa, sigma, rho)
    volswap_numint[i] = numerical_volswap_price(K, r, q, volswap_maturity, timestep, S0, v0, theta, kappa, sigma, rho)

# now Print results
print("VarSwap and VolSwap Fair Values:")



# we Generate the volatility surface using the Heston model
strikes = np.arange(50, 151, 10)
expiries = np.arange(1, 11)
vols = np.zeros((len(expiries), len(strikes)))
for i in range(len(expiries)):
    for j in range(len(strikes)):
        vols[i][j] = heston_model.implied_vol(strikes[j], expiries[i])

# we Calculate the fair value of a VarSwap
varstrike = 0.05
timesteps = 100
n_simulations = 100000
dt = varswap_expiry / timesteps
varswap_mc = MonteCarloVarSwap(heston_model, varstrike, varswap_expiry, timesteps, n_simulations)
fair_varswap, se_varswap = varswap_mc.run_simulation()

# Now Applying Antithetic Variance Reduction
varswap_mc_avr = MonteCarloVarSwapAVR(heston_model, varstrike, varswap_expiry, timesteps, n_simulations)
fair_varswap_avr, se_varswap_avr = varswap_mc_avr.run_simulation()

#Then Calculating the fair value of a VolSwap
volstrike = 0.2
timesteps = 100
n_simulations = 100000
dt = volswap_expiry / timesteps
volswap_mc = MonteCarloVolSwap(heston_model, volstrike, volswap_expiry, timesteps, n_simulations)
fair_volswap, se_volswap = volswap_mc.run_simulation()

# Applying Control Variate Variance Reduction
volswap_mc_cv = MonteCarloVolSwapCV(heston_model, volstrike, volswap_expiry, timesteps, n_simulations)
fair_volswap_cv, se_volswap_cv = volswap_mc_cv.run_simulation()

# Applying Stratified Sampling
volswap_mc_ss = MonteCarloVolSwapSS(heston_model, volstrike, volswap_expiry, timesteps, n_simulations)
fair_volswap_ss, se_volswap_ss = volswap_mc_ss.run_simulation()

# Printing the results
print(f"Underlying Price: {spot_price}")
print(f"Heston Parameters: {heston_params}")
print(f"VarSwap Expiry: {varswap_expiry}, VarSwap Strike: {varstrike}")
print(f"Monte Carlo Fair Value of VarSwap: {fair_varswap:.4f}, Standard Error: {se_varswap:.4f}")
print(f"Monte Carlo (AVR) Fair Value of VarSwap: {fair_varswap_avr:.4f}, Standard Error: {se_varswap_avr:.4f}")
print(f"VolSwap Expiry: {volswap_expiry}, VolSwap Strike: {volstrike}")
print(f"Monte Carlo Fair Value of VolSwap: {fair_volswap:.4f}, Standard Error: {se_volswap:.4f}")
print(f"Monte Carlo (CV) Fair Value of VolSwap: {fair_volswap_cv:.4f}, Standard Error: {se_volswap_cv:.4f}")
print(f"Monte Carlo (SS) Fair Value of VolSwap: {fair_volswap_ss:.4f}, Standard Error: {se_volswap_ss:.4f}")

# Convergence tests (to be completed)
n_sims = [1000, 10000, 100000, 1000000]
timesteps = 100
varswap_results = []
volswap_results = []
for
