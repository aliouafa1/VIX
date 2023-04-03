import financepy as fp
import numpy as np

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
sabr_parameters = fp.SABRParameters(beta=0.7, rho=-0.3, nu=0.5, volvol=0.5)

# Set up volatility surface parameters
strikes = np.arange(50, 151, 5)
expiries = np.arange(0.25, 2.51, 0.25)

# Calculate fair values and compare with ATMI
print("*** SABR Model Results ***")

# Calculate VarSwap fair value
varswap_fair_value = fp.equity.varswap.priceVarSwap(
    spot,
    varswap_start_date,
    varswap_end_date,
    strikes,
    atm_volatility,
    risk_free_rate,
    dividend_yield,
    sabr_parameters,
    varswap_fixed_rate,
)
print("VarSwap fair value: {:.4f}".format(varswap_fair_value))
if varswap_fair_value < atmi_price:
    print("Is VarSwap cheaper than ATMI? Yes")
else:
    print("Is VarSwap cheaper than ATMI? No")

# Calculate VolSwap fair value
volswap_fair_value, vol_surface = fp.equity.volswap.priceVolSwap(
    spot,
    volswap_start_date,
    volswap_end_date,
    expiries,
    strikes,
    atm_volatility,
    risk_free_rate,
    dividend_yield,
    sabr_parameters,
    volswap_fixed_rate,
    vol_swap_type=fp.VolSwapType.VOLATILITY,
    return_surface=True,
)
print("VolSwap fair value: {:.4f}".format(volswap_fair_value))
if volswap_fair_value < atmi_price:
    print("Is VolSwap cheaper than ATMI? Yes")
else:
    print("Is VolSwap cheaper than ATMI? No")

# Print volatility surfaces
print("\n*** Volatility Surfaces ***")
print("SABR model:")
for i, expiry in enumerate(expiries):
    print("Expiry: {:.2f} years".format(expiry))
    for j, strike in enumerate(strikes):
        print("{:.2f}  {:.4f}".format(strike, vol_surface[i, j]))
    print("")
