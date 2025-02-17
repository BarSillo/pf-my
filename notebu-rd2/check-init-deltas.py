# check deltas:

import numpy as np
from scipy.stats import norm

# Parameters (annualized perspective where needed)
K = 100              # Strike price
r = 0.05             # Risk-free rate (annual) used for Black-Scholes
mu = 0.07            # Expected stock return (not directly used for Delta)
daily_vol = 0.01     # Daily volatility = 1%
sigma = daily_vol * np.sqrt(252)  # Convert to annualized volatility
days_to_maturity = 1.1
T = days_to_maturity / 252.0      # Time to maturity in years



# Desired initial deltas
desired_deltas = [0.20, 0.40, 0.50, 0.60, 0.80]

def find_initial_stock_price_for_delta(delta, K, r, sigma, T):
    """
    Invert the Delta formula (for a European call) to get the stock price S0
    that yields a target Delta using the Black-Scholes 'd1' relationship:

       Delta = N(d1)
       d1 = [ln(S0/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))

    Solve for S0 given 'delta'.
    """
    # Inverse standard normal
    z = norm.ppf(delta)

    # Solve for S0:
    # log(S0/K) = z*sigma*sqrt(T) - (r + 0.5*sigma^2)*T
    # S0 = K * exp( ... )
    lhs = z * sigma * np.sqrt(T) - (r + 0.5 * sigma**2) * T
    S0 = K * np.exp(lhs)
    return S0

# Compute and display initial stock prices for each desired delta
print("Daily stock volatility:", f"{daily_vol*100:.2f}% (annualized ~ {sigma*100:.2f}%)")
print(f"Time to maturity: {days_to_maturity} days (~{T:.4f} years)\n")

for d in desired_deltas:
    S0_for_delta = find_initial_stock_price_for_delta(d, K, r, sigma, T)
    print(f"Target Delta = {d*100:.0f}%, Initial Stock Price S0 = {S0_for_delta:.2f}")

print(find_initial_stock_price_for_delta(np.array(desired_deltas), K, r, sigma, T ))