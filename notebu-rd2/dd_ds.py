"""
dd-ds-cached.py - Delta Difference Simulation using Black-Scholes
with cached random normals.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Tuple
from functools import cached_property


class BSOption:
    """Base class for Black-Scholes option calculations"""
    def __init__(self, K: float, r: float, sigma: float, T: float):
        """
        Initialize Black-Scholes parameters.
        
        Args:
            K: Strike price
            r: Risk-free rate (annualized)
            sigma: Volatility (annualized)
            T: Time to expiration (years)
        """
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    def _d1(self, S: float, t: float) -> float:
        """Calculate d1 parameter"""
        return (np.log(S/self.K) + (self.r + 0.5*self.sigma**2)*t) / (self.sigma*np.sqrt(t))

    def delta(self, S: float, t: float) -> float:
        """Calculate option delta"""
        return norm.cdf(self._d1(S, t))

    def theta(self, S: float, t: float) -> float:
        """Calculate option theta"""
        if t <= 1e-6:
            return 0.0
        d1 = self._d1(S, t)
        d2 = d1 - self.sigma*np.sqrt(t)
        return - (S*self.sigma*norm.pdf(d1))/(2*np.sqrt(t)) - self.r*self.K*np.exp(-self.r*t)*norm.cdf(d2)

    def charm(self, S: float, t: float) -> float:
        """Calculate option charm (delta decay)"""
        if t <= 1e-6:
            return 0.0
        d1 = self._d1(S, t)
        pdf = norm.pdf(d1)
        term1 = (self.r + 0.5*self.sigma**2)/(self.sigma*np.sqrt(t))
        term2 = d1/(2*t)
        return -pdf * (term1 - term2)


class DeltaSimulator(BSOption):
    """Simulates delta differences under price moves"""
    def __init__(self, K: float, r: float, sigma: float, T: float,
                 delta_t: float, mu: float = 0.07, N_MC: int = 100_000):
        """
        Initialize simulator.
        
        Args:
            delta_t: Simulation timestep (years)
            mu: Real-world drift parameter
            N_MC: Number of Monte Carlo paths (for generating random normals)
        """
        super().__init__(K, r, sigma, T)
        self.delta_t = delta_t
        self.mu = mu
        self.sigma_d = sigma * np.sqrt(delta_t)
        self.N_MC = N_MC

    @cached_property
    def RN(self) -> np.ndarray:
        """
        Cached random normals array generated on first access.
        
        Returns:
            np.ndarray: Array of random normals with shape (N_MC,)
        """
        return np.random.randn(self.N_MC)

    def _implied_S0(self, initial_delta: float) -> float:
        """Calculate initial stock price implied from delta"""
        d1_value = norm.ppf(initial_delta)
        return self.K * np.exp(
            d1_value * self.sigma * np.sqrt(self.T) -
            (self.r + 0.5*self.sigma**2)*self.T
        )

    def simulate(self, initial_delta: float) -> pd.Series:
        """
        Simulate delta differences across price move thresholds.
        
        Args:
            initial_delta: Target initial delta value
            
        Returns:
            pd.Series: Mean delta differences indexed by price deviation
        """
        S0 = self._implied_S0(initial_delta)
        print(f"Implied initial S0 from Δ0={initial_delta:.2f}: {S0:.3f}")

        # Generate price paths using the cached random normals.
        S1 = S0 * np.exp(
            (self.mu - 0.5*self.sigma**2)*self.delta_t +
            self.sigma*np.sqrt(self.delta_t)*self.RN
        )

        # Calculate delta differences.
        D0 = self.delta(S0, self.T)
        D1 = self.delta(S1, self.T - self.delta_t)
        delta_diff = D1 - D0

        # Define sigma-level thresholds.
        sigma_levels = np.arange(-6, 6.05, 0.1)
        mask_neg = sigma_levels < 0
        mask_pos = sigma_levels > 0

        thresholds_neg = S0 * (1 + sigma_levels[mask_neg] * self.sigma_d)
        thresholds_pos = S0 * (1 + sigma_levels[mask_pos] * self.sigma_d)
        all_thresholds = np.concatenate([thresholds_neg, thresholds_pos])

        # Vectorized boundary checks.
        S1_expanded = S1.reshape(1, -1)
        neg_masks = (S1_expanded >= thresholds_neg.reshape(-1, 1)) & (S1 < S0)
        pos_masks = (S1_expanded <= thresholds_pos.reshape(-1, 1)) & (S1 > S0)
        all_masks = np.vstack([neg_masks, pos_masks])

        # Calculate means for each threshold.
        means = [
            delta_diff[mask_row].mean() if mask_row.any() else np.nan
            for mask_row in all_masks
        ]
        res = pd.Series(
            means,
            index=all_thresholds - S0,
            name='Δ Difference'
        ).sort_index().dropna()
        return res

    def theta_decay(self, S: float) -> Tuple[float, float]:
        """Calculate theta decay components"""
        theta = self.theta(S, self.T)
        return theta, theta * self.delta_t

    def charm_decay(self, S: float) -> Tuple[float, float]:
        """Calculate charm decay components"""
        chrm = self.charm(S, self.T)
        return chrm, chrm * self.delta_t


if __name__ == "__main__":
    # Create the simulator with a fixed number of Monte Carlo paths.
    simulator = DeltaSimulator(
        K=100,
        r=0.05,
        sigma=1/100.0 * np.sqrt(252),
        T=5/252,
        delta_t=1/252,
        mu=0.07,
        N_MC=100_000
    )

    # Example: Run simulation for multiple initial delta values.
    initial_deltas = [0.2, 0.4, 0.5, 0.6, 0.8]
    simulation_results = {}

    for delta_val in initial_deltas:
        results = simulator.simulate(initial_delta=delta_val)
        simulation_results[delta_val] = results
        print(f"\nResults for initial delta {delta_val}:\n", results)

        # Plot each result.
        results.plot(style='o-', title=f'Δ Difference vs Price Deviation (Δ₀={delta_val})')
        plt.xlabel('Price Deviation from S0')
        plt.ylabel('Mean Δ Difference')
        plt.grid(True)
        plt.show()

    # Theta decay example.
    S0_example = 100
    theta, theta_decay = simulator.theta_decay(S0_example)
    print(f"\nTheta: {theta:.4f}/yr, Daily Decay: {theta_decay:.6f}")

    # Charm decay example.
    charm, charm_effect = simulator.charm_decay(S0_example)
    print(f"Charm: {charm:.4f}/yr, Daily Δ Change: {charm_effect:.6f}")
