import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
r = 0.05       # Risk-free rate
T = 1          # Time to maturity (in years)
volatility_range = np.linspace(0.1, 0.5, 50)  # Volatility range from 10% to 50%
log_moneyness_range = np.linspace(-0.5, 0.5, 50)  # Log-moneyness range from -0.5 to 0.5

# Function to compute the realized delta using the revised formulation
def realized_delta_volatility_units(volatility, log_moneyness):
    S = 1  # Normalize stock price to 1 for log-moneyness calculations
    K = S * np.exp(-log_moneyness)  # Rescale strike price based on log-moneyness
    S_low = S * np.exp(-3 * volatility)  # Lower bound: -3 sigma move
    S_high = S * np.exp(3 * volatility)  # Upper bound: +3 sigma move
    
    def d1(S_prime):
        return (np.log(S_prime / K) + (r + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
    
    def delta(S_prime):
        return norm.cdf(d1(S_prime))
    
    def lognormal_pdf(S_prime):
        exponent = -((np.log(S_prime / S) - (r - 0.5 * volatility ** 2) * T) ** 2) / (2 * volatility ** 2 * T)
        return (1 / (S_prime * volatility * np.sqrt(2 * np.pi * T))) * np.exp(exponent)
    
    def integrand(S_prime):
        return delta(S_prime) * lognormal_pdf(S_prime)
    
    result, _ = quad(integrand, S_low, S_high)
    
    # Normalize by the size of the 3-sigma range to express in volatility units
    return result / (3 * volatility)

# Compute realized delta changes for different combinations of volatility and log-moneyness
delta_changes_volatility_units = np.zeros((len(volatility_range), len(log_moneyness_range)))

for i, vol in enumerate(volatility_range):
    for j, log_m in enumerate(log_moneyness_range):
        delta_changes_volatility_units[i, j] = realized_delta_volatility_units(vol, log_m)

# Plotting the surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(log_moneyness_range, volatility_range)
ax.plot_surface(X, Y, delta_changes_volatility_units, cmap='viridis')

ax.set_xlabel('Log-Moneyness (log(S/K))')
ax.set_ylabel('Volatility')
ax.set_zlabel('Realized Delta (in Volatility Units)')
ax.set_title('Surface Plot of Realized Delta in Volatility Units')

plt.show()
