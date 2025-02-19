import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from functools import cached_property

# Update pandas display options
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 250)

class BSOption:
    """
    Basic Black-Scholes option pricing and Greeks.
    """
    def __init__(self, K: float, r: float, sigma: float, T: float):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        
    def _d1(self, S: float, t: float) -> float:
        return (np.log(S/self.K) + (self.r + 0.5 * self.sigma**2)*t) / (self.sigma * np.sqrt(t))
    
    def delta(self, S: float, t: float) -> float:
        return norm.cdf(self._d1(S, t))
    
    def gamma(self, S: float, t: float) -> float:
        d1 = self._d1(S, t)
        return norm.pdf(d1) / (S * self.sigma * np.sqrt(t))
    
    def charm(self, S: float, t: float) -> float:
        """
        Option charm: the sensitivity of delta to the passage of time.
        """
        if t <= 1e-6:
            return 0.0
        d1 = self._d1(S, t)
        pdf = norm.pdf(d1)
        term1 = (self.r + 0.5 * self.sigma**2) / (self.sigma * np.sqrt(t))
        term2 = d1 / (2 * t)
        return -pdf * (term1 - term2)

class DeltaSimulator(BSOption):
    """Simulates delta differences under price moves using Monte Carlo."""
    def __init__(self, K: float, r: float, sigma: float, T: float, delta_t: float, mu: float = 0.07, N_MC: int = 100000):
        super().__init__(K, r, sigma, T)
        self.delta_t = delta_t  # one-day time step
        self.mu = mu
        self.sigma_d = sigma * np.sqrt(delta_t)
        self.N_MC = N_MC
        
    @cached_property
    def RN(self) -> np.ndarray:
        return np.random.randn(self.N_MC)
    
    @property
    def default_canvas(self):
        """
        Returns a styled (fig, ax) tuple:
          - Figure size: 10 x 6 inches.
          - Figure background: light beige (#F5F5DC).
          - Axes background: lighter beige (#FAF0E6).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#F5F5DC')
        ax.set_facecolor('#FAF0E6')
        return fig, ax
    
    def _implied_S0(self, initial_delta: float) -> float:
        """
        Calculate the implied initial stock price S₀ from the given initial delta.
        """
        d1_value = norm.ppf(initial_delta)
        return self.K * np.exp(d1_value * self.sigma * np.sqrt(self.T) - (self.r + 0.5*self.sigma**2)*self.T)
    
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
    
    def simulate_multiple(self, initial_deltas: list) -> pd.DataFrame:
        """
        For each initial delta, run simulation and interpolate results onto a common grid.
        Returns a DataFrame with one column per initial delta.
        """
        sim_results = {}
        for d in initial_deltas:
            series = self.simulate(d)
            key = f"Δ₀={d} (S₀={self._implied_S0(d):.2f})"
            sim_results[key] = series
        all_indices = np.concatenate([s.index.values for s in sim_results.values()])
        global_min = all_indices.min()
        global_max = all_indices.max()
        common_grid = np.linspace(global_min, global_max, 200)
        interpolated = {}
        for key, series in sim_results.items():
            new_vals = np.interp(common_grid, series.index.values, series.values)
            interpolated[key] = pd.Series(new_vals, index=common_grid, name=key)
        return pd.DataFrame(interpolated)
    
    def initial_charm_change(self, initial_delta: float) -> float:
        """
        Compute the one-day change in delta due solely to option charm at the option's initial S₀.
        Returns: Δ_charm = charm(S₀, T - delta_t) * delta_t.
        """
        S0 = self._implied_S0(initial_delta)
        return self.charm(S0, self.T - self.delta_t) * self.delta_t
    

# -----------------------------
# MAIN SECTION
# -----------------------------
if __name__ == "__main__":
    # Initial parameters:
    params = {
        'K': 100,           # Strike price
        'r': 0.05,          # Risk-free rate
        'sigma': 0.158,     # Annual volatility
        'T': 5.0 / 252,     # Time to expiration (~5 trading days)
        'mu': 0.07,         # Real-world drift
        'delta_t': 1/252,   # One-day time step
        'N_MC': 100000
    }
    
    simulator = DeltaSimulator(**params)
    initial_deltas = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    
    # 1. Simulated Delta Differences vs Price Deviation
    df_sim = simulator.simulate_multiple(initial_deltas)
    fig, ax = simulator.default_canvas
    df_sim.plot(ax=ax)
    ax.set_xlabel("Price Deviation from S₀")
    ax.set_ylabel("Simulated Δ Difference")
    ax.set_title("Simulated Delta Differences vs Price Deviation")
    ax.grid(True, axis='y')
    plt.show()
    

        # 2. Gamma-Charm Approximation vs Simulation at 5-sigma Move
    # For each initial delta, extract the simulated Δ difference at the 5-sigma move
    # and compute the gamma-charm approximation at that move.
    results_data = []
    for d in initial_deltas:
        S0 = simulator._implied_S0(d)
        sigma_d = simulator.sigma * np.sqrt(simulator.delta_t)
        dS_5sigma = 5 * sigma_d  # 5-sigma move
        
        # Run simulation for this initial delta:
        sim_series = simulator.simulate(d)
        # Find the simulated delta difference at the index closest to dS_5sigma
        idx_closest = np.abs(sim_series.index - dS_5sigma).argmin()
        sim_value = sim_series.iloc[idx_closest]
        
        # Compute gamma-charm approximation at S₀:
        gamma_val = simulator.gamma(S0, simulator.T)
        charm_val = simulator.charm(S0, simulator.T - simulator.delta_t)
        approx_value = gamma_val * dS_5sigma + charm_val * simulator.delta_t
        diff = sim_value - approx_value
        
        results_data.append({
            "Initial Delta": d,
            "S₀": S0,
            "5-Sigma Move": dS_5sigma,
            "Simulated Δ Change": sim_value,
            "Gamma-Charm Approx": approx_value,
            "Difference": diff
        })
    
    df_5sigma = pd.DataFrame(results_data).set_index("Initial Delta")
    print("\nGamma-Charm vs Simulation at 5-sigma Move:")
    print(df_5sigma)
    
    # Plot a bar chart comparing simulation and gamma-charm approximation at 5-sigma move
    fig, ax = simulator.default_canvas
    width = 0.35
    x = np.arange(len(df_5sigma))
    ax.bar(x - width/2, df_5sigma["Simulated Δ Change"], width, label="Simulated Δ Change", color='orchid')
    ax.bar(x + width/2, df_5sigma["Gamma-Charm Approx"], width, label="Gamma-Charm Approx", color='orange')
    ax.set_xlabel("Initial Delta")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.2f}" for d in df_5sigma.index])
    ax.set_ylabel("Δ Change")
    ax.set_title("Gamma-Charm Approximation vs Simulation at 5-sigma Move")
    ax.legend()
    ax.grid(True, axis='y')
    plt.show()

        # 2. Summary of Initial Greeks and Charm Effect
    summary_df = simulator.summary_initial_greeks(initial_deltas)
    print("\nSummary of Initial Greeks and Charm Effect:")
    print(summary_df)
    fig, ax = simulator.default_canvas
    width = 0.25
    x = np.arange(len(summary_df))
    ax.bar(x - width, summary_df["Calculated Delta"], width, label="Initial Delta", color='lightgreen')
    ax.bar(x, summary_df["Initial Gamma"], width, label="Initial Gamma", color='salmon')
    ax.bar(x + width, summary_df["Initial Charm"], width, label="Initial Charm", color='skyblue')
    ax.set_xlabel("Initial Delta")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df.index.round(2))
    ax.set_ylabel("Value")
    ax.set_title("Summary of Initial Delta, Gamma, and One-Day Charm Effect")
    ax.legend()
    ax.grid(True, axis='y')
    plt.show()

    
    # 3. Simulated Delta Differences vs Price Deviation (Normalized by Gamma)
    normalized_df = df_sim.copy()
    for d in initial_deltas:
        key = f"Δ₀={d} (S₀={simulator._implied_S0(d):.2f})"
        gamma_val = simulator.gamma(simulator._implied_S0(d), simulator.T)
        normalized_df[key] = normalized_df[key] / gamma_val
    fig, ax = simulator.default_canvas
    normalized_df.plot(ax=ax)
    ax.set_xlabel("Price Deviation from S₀")
    ax.set_ylabel("Normalized Simulated Δ Difference (by Gamma)")
    ax.set_title("Simulated Delta Differences vs Price Deviation (Normalized by Gamma)")
    ax.grid(True, axis='y')
    plt.show()
    
    # 4. Gamma-Charm Approximation Summary for Initial Deltas (up to 5-sigma one-day move)
    simulator.plot_gamma_charm_approximation(initial_deltas)
