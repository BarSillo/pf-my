from scipy.integrate import quad
from scipy.stats import lognorm, norm
import numpy as np


def expected_delta_diff(n_sigma, r,  K,  Delta0, mu, sigma, T, delta_t):
    # Parameters for log-normal distribution
    drift = (mu - 0.5*sigma**2) * delta_t
    scale = sigma * np.sqrt(delta_t)
    
    S0 =  K * np.exp(
            Delta0 * sigma * np.sqrt(T) - (r + 0.5 * sigma**2) * T)
    
    # PDF of S1
    def pdf(S):
        z = (np.log(S/S0) - drift) / scale
        return norm.pdf(z) / (S * scale)
    
    # Integrand: (Delta(S) - Delta0) * f(S)
    def integrand(S):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*(T - delta_t)) / \
             (sigma * np.sqrt(T - delta_t))
        Delta_S = norm.cdf(d1)
        return (Delta_S - Delta0) * pdf(S)
    
    # Compute numerator and denominator
    L_H_tuple = S0, S0 + n_sigma * sigma * S0 * np.sqrt(delta_t )
    S_low, S_high = sorted(L_H_tuple)    
    numerator, _ = quad(integrand, S_low, S_high)
    denominator, _ = quad(pdf, S_low, S_high)
    
    return numerator / denominator if denominator != 0 else np.nan


if __name__ == "__main__":
    print("expected flow: {}".format(
        expected_delta_diff(n_sigma=5, r=0.05,  K=100,  Delta0=0.8, mu=0.07, sigma=0.158, T=5/252, delta_t=1/252)
    ))