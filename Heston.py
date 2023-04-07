# Generating the volatility surface using the Heston model
strikes = np.arange(50, 151, 10)
expiries = np.arange(1, 11)
vols = np.zeros((len(expiries), len(strikes)))
for i in range(len(expiries)):
    for j in range(len(strikes)):
        vols[i][j] = heston_model.implied_vol(strikes[j], expiries[i])

# Calculating the fair value of a VarSwap
varstrike = 0.05
timesteps = 100
n_simulations = 100000
dt = varswap_expiry / timesteps
varswap_mc = MonteCarloVarSwap(heston_model, varstrike, varswap_expiry, timesteps, n_simulations)
fair_varswap, se_varswap = varswap_mc.run_simulation()

# Applying Antithetic Variance Reduction
varswap_mc_avr = MonteCarloVarSwapAVR(heston_model, varstrike, varswap_expiry, timesteps, n_simulations)
fair_varswap_avr, se_varswap_avr = varswap_mc_avr.run_simulation()

# Calculating the fair value of a VolSwap
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
