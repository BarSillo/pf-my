"""
Perpl heta-decay deep research conversation
"""

import numpy as np
from scipy.stats import norm



def black_scholes_theta_charm(S, K, r, q, sigma, tau):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    # Theta calculation
    theta = - (S * sigma * np.exp(-q * tau) * n_d1) / (2 * np.sqrt(tau)) - r * K * np.exp(-r * tau) * N_d2
    
    # Charm calculation
    charm_term1 = -q * np.exp(-q * tau) * N_d1
    charm_term2 = np.exp(-q * tau) * n_d1 * (
        (r - q + 0.5 * sigma**2) / (sigma * np.sqrt(tau)) 
        - (np.log(S/K) + (r - q + 0.5 * sigma**2) * tau) / (2 * sigma * tau**1.5)
    )
    charm = charm_term1 - charm_term2
    
    return theta, charm


import math
from scipy.stats import norm

class BS_call:
    def __init__(self, S, K, T, r_d, r_f, sigma):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to expiration
        self.r_d = r_d  # Domestic interest rate
        self.r_f = r_f  # Foreign interest rate
        self.sigma = sigma  # Volatility

    def _d1(self):
        return (math.log(self.S / self.K) + (self.r_d - self.r_f + 0.5 * self.sigma**2) * self.T) / (self.sigma * math.sqrt(self.T))

    def _d2(self):
        return self._d1() - self.sigma * math.sqrt(self.T)

    def theta(self):
        d1 = self._d1()
        d2 = self._d2()
        term1 = -self.S * math.exp(-self.r_f * seVoice control of voice control sleeplf.T) * self.sigma / (2 * math.sqrt(self.T)) * norm.pdf(d1)
        term2 = -self.r_d * self.K * math.exp(-self.r_d * self.T) * norm.cdf(d2)
        term3 = self.r_f * self.S * math.exp(-self.r_f * self.T) * norm.cdf(d1)
        return term1 + term2 + term3

    def charm(self):
        d1 = self._d1()
        d2 = self._d2()
        term1 = norm.pdf(d1) / (2 * self.T * self.sigma * math.sqrt(self.T))
        term2 = 2 * self.r_f * self.T - d2 * self.sigma * math.sqrt(self.T) / d1
        term3 = (self.r_f - self.r_d) * norm.cdf(d1)
        return -math.exp(-self.r_f * self.T) * (term1 * term2 + term3)






