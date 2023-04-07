import numpy as np
from scipy.stats import norm

# Set up market data
spot = 100.0
risk_free_rate = 0.01
dividend_yield = 0.0
atm_volatility = 0.20

# Set up VarSwap parameters
varswap_start_date = "03-Apr-2022"
varswap_end_date = "03-Apr-2023"
varswap_fixed_rate = 0.02

# Set up VolSwap parameters
volswap_start_date = "03-Apr-2022"
volswap_end_date = "03-Apr-2023"
volswap_fixed_rate = 0.03

# Set up SABR model parameters
beta = 0.7
rho = -0.3
nu = 0.5
volvol = 0.5

# Set up volatility surface parameters
strikes = np.arange(50, 151, 5)
expiries = np.arange(0.25, 2.51, 0.25)

# Define the SABR model function
def sabr_volatility(strike, expiry, spot, beta, rho, nu, volvol):
    forward = spot * np.exp((risk_free_rate - dividend_yield) * expiry)
    alpha = forward ** (1 - beta)
    z = nu / alpha * (forward * strike) ** ((1 - beta) / 2) * np.log(forward / strike)
    x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    factor1 = alpha / (forward * strike) ** ((1 - beta) / 2)
    factor2 = 1 + ((1 - beta) ** 2 / 24 * np.log(forward / strike) ** 2 + (1 - beta) ** 4 / 1920 * np.log(forward / strike) ** 4) * alpha ** 2 / (forward * strike) ** (1 - beta)
    factor3 = nu / alpha * x
    return factor1 * factor2 * factor3

# Calcul of the fair values and comparison with ATMI
print("*** SABR Model Results ***")

# Calculate VarSwap fair value
varswap_fair_value = 0.0
for i in range(len(strikes)):
    strike = strikes[i]
    vol = sabr_volatility(strike, (varswap_end_date - varswap_start_date) / 365, spot, beta, rho, nu, volvol)
    d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + vol ** 2 / 2) * (varswap_end_date - varswap_start_date) / 365) / (vol * np.sqrt((varswap_end_date - varswap_start_date) / 365))
    d2 = d1 - vol * np.sqrt((varswap_end_date - varswap_start_date) / 365)
    varswap_fair_value += 2 * np.sqrt(varswap_start_date * varswap_end_date) * (strike * np.exp(-dividend_yield * (varswap_end_date - varswap_start_date) / 365) * norm.cdf(-d2) - spot * norm.cdf(-d1)) * norm.pdf(d1) * vol / np.sqrt((varswap_end_date - varswap_start_date) / 365)
varswap_fair_value *= np.exp(-risk_free_rate * (varswap_end_date - varswap_start_date) / 365)


# Calculate VolSwap fair value
volswap_fair_value = 0.0
for i in range(len(strikes)):
strike = strikes[i]
vol = sabr_volatility(strike, (volswap_end_date - volswap_start_date) / 365, spot, beta, rho, nu, volvol)
d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + vol ** 2 / 2) * (volswap_end_date - volswap_start_date) / 365) / (vol * np.sqrt((volswap_end_date - volswap_start_date) / 365))
d2 = d1 - vol * np.sqrt((volswap_end_date - volswap_start_date) / 365)
volswap_fair_value += 2 * np.sqrt(volswap_start_date * volswap_end_date) * (spot * np.exp((risk_free_rate - dividend_yield) * (volswap_end_date - volswap_start_date) / 365) * norm.cdf(d1) - strike * np.exp(-risk_free_rate * (volswap_end_date - volswap_start_date) / 365) * norm.cdf(d2)) * norm.pdf(d1) * vol / np.sqrt((volswap_end_date - volswap_start_date) / 365)
volswap_fair_value *= np.exp(-risk_free_rate * (volswap_end_date - volswap_start_date))

# Comparison with ATMI
print("ATMI: ", atm_volatility)
print("VarSwap Fair Value: ", varswap_fair_value)
print("VolSwap Fair Value: ", volswap_fair_value)
