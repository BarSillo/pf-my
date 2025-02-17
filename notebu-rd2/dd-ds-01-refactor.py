"""
This is deep-seek formatted and I have a question regarding the range of delta difference.
good version thats working is at dd_ds.py this file is for reference
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from typing import Tuple

class BSOption:
    def __init__(self, K: float, r: float, sigma: float, T: float, delta_t: float, mu: float = 0.07):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.delta_t = delta_t
        self.mu = mu
        self.sigma_d = sigma * np.sqrt(delta_t)

    @staticmethod
    def _calculate_d1(S: float, K: float, r: float, sigma: float, t: float) -> float:
        return (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))

    def _calculate_delta(self, S: float, t: float) -> float:
        d1_val = self._calculate_d1(S, self.K, self.r, self.sigma, t)
        return norm.cdf(d1_val)

    def _calculate_theta(self, S: float, t: float) -> float:
        """Calculate theta (time decay) for European call option."""
        if t <= 1e-6:
            return 0.0
        
        d1 = self._calculate_d1(S, self.K, self.r, self.sigma, t)
        d2 = d1 - self.sigma * np.sqrt(t)
        
        theta = (
            - (S * self.sigma * norm.pdf(d1)) / (2 * np.sqrt(t))
            - self.r * self.K * np.exp(-self.r * t) * norm.cdf(d2)
        )
        return theta

    def _calculate_charm(self, S: float, t: float) -> float:
        """Calculate the Charm (delta decay) for European call option."""
        if t <= 1e-6:
            return 0.0
        
        d1 = self._calculate_d1(S, self.K, self.r, self.sigma, t)
        tau = t
        
        n_d1 = norm.pdf(d1)
        term1 = (self.r + 0.5 * self.sigma**2) / (self.sigma * np.sqrt(tau))
        term2 = d1 / (2 * tau)
        
        return -n_d1 * (term1 - term2)


class OptionDeltaSimulator(BSOption):
    def simulate(self, initial_delta: float, N_MC: int = 100_000) -> pd.Series:
        d1_value = norm.ppf(initial_delta)
        S0 = self.K * np.exp(
            d1_value * self.sigma * np.sqrt(self.T)
            - (self.r + 0.5 * self.sigma**2) * self.T
        )
        print(f"Implied initial S0 from Δ0={initial_delta:.2f}: {S0:.3f}")

        RN = np.random.randn(N_MC, 1)
        S1 = S0 * np.exp(
            (self.mu - 0.5 * self.sigma**2) * self.delta_t
            + self.sigma * np.sqrt(self.delta_t) * RN
        ).squeeze()

        D0 = self._calculate_delta(S0, self.T)
        D1 = self._calculate_delta(S1, self.T - self.delta_t)
        delta_diff = D1 - D0

        sigma_thresholds = np.arange(-6, 6.05, 0.1)
        price_thresholds = S0 * (1 + sigma_thresholds * self.sigma_d)

        S1_expanded = S1.reshape(1, -1)
        thresholds_expanded = price_thresholds.reshape(-1, 1)
        
        neg_masks = (S1_expanded >= thresholds_expanded[sigma_thresholds < 0]) & (S1_expanded < S0)
        pos_masks = (S1_expanded <= thresholds_expanded[sigma_thresholds > 0]) & (S1_expanded > S0)

        all_masks = np.vstack([neg_masks, pos_masks])
        means = np.array([delta_diff[mask].mean() if np.any(mask) else np.nan for mask in all_masks])

        out = pd.Series(means, index=price_thresholds - S0, name='Delta Difference').sort_index().dropna()

        return out


class OptionDeltaCalc(BSOption):
    def calculate_theta_decay(self, S: float) -> Tuple[float, float]:
        theta_instant = self._calculate_theta(S, self.T)
        theta_decay = theta_instant * self.delta_t
        return theta_instant, theta_decay

    def calculate_charm(self, S: float, t: float) -> float:
        return self._calculate_charm(S, t)

    def calculate_charm_decay(self, S: float, t: float) -> Tuple[float, float]:
        charm_instant = self._calculate_charm(S, t)
        delta_change = charm_instant * self.delta_t
        return charm_instant, delta_change


if __name__ == "__main__":
    simulator = OptionDeltaCalc(K=100, r=0.05, sigma=1/100.0 * np.sqrt(252), T=5/252, delta_t=1/252)
    
    S0 = 100  
    theta_instant, theta_decay = simulator.calculate_theta_decay(S0)
    
    print(f"Instantaneous theta: {theta_instant:.4f} per year")
    print(f"Daily theta decay: {theta_decay:.6f} (absolute value)")
    print(f"Equivalent daily decay: {theta_decay/S0:.4%} of spot")

    days = np.linspace(1/252, 30/252, 30)
    thetas = [simulator._calculate_theta(S0, t) for t in days]
    
    plt.plot(days*252, thetas)
    plt.xlabel('Days to Expiration')
    plt.ylabel('Theta (per year)')
    plt.title('Time Decay Profile (S=K=100, σ=15%, r=5%)')
    plt.grid(True)
    plt.show()
