import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt
from dd_ds import BSOption


class DeltaCalculator(BSOption):
    """Numerical integration–based delta change calculator and charm evaluator."""
    def __init__(self, K: float, r: float, sigma: float, T: float, mu: float = 0.07):
        """
        Args:
            mu: Real-world drift parameter.
        """
        super().__init__(K, r, sigma, T)
        self.mu = mu
        self._wopts = None  # Optional: parameters for quad optimization

    @property
    def default_canvas(self):
        """
        Returns a tuple (fig, ax) with default canvas settings:
         - Figure size: 10 x 6 inches.
         - Figure background: light beige (#F5F5DC).
         - Axes background: lighter beige (#FAF0E6).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#F5F5DC')
        ax.set_facecolor('#FAF0E6')
        return fig, ax

    def _log_normal_pdf(self, S: float, S0: float, t: float) -> float:
        """
        Log-normal probability density function for S given initial S0 and time t.
        """
        drift = (self.mu - 0.5 * self.sigma**2) * t
        scale = self.sigma * np.sqrt(t)
        z = (np.log(S / S0) - drift) / scale
        return norm.pdf(z) / (S * scale)

    def _delta_diff_integrand(self, S: float, S0: float, t: float) -> float:
        """
        Integrand for the expected delta change:
        (Δ(S) - Δ(S0)) * PDF(S).
        """
        return (self.delta(S, self.T - t) - self.delta(S0, self.T)) * self._log_normal_pdf(S, S0, t)

    def expected_delta_changes(self, S0: float, sigma_moves: np.ndarray, delta_t: float) -> pd.Series:
        """
        For a given starting price S0 and a range of sigma moves, compute the expected delta change
        conditioned on the move reaching the boundary:
            S_b = S0 * (1 + sigma_move * sigma * sqrt(delta_t))
        
        Returns a pd.Series indexed by (S_b - S0) with the computed expected delta difference.
        """
        results = {}
        t = delta_t
        sigma_d = self.sigma * np.sqrt(t)
        
        for sigma_move in sigma_moves:
            boundary = S0 * (1 + sigma_move * sigma_d)
            a, b = sorted([S0, boundary])
            
            num, _ = quad(self._delta_diff_integrand, a, b, args=(S0, t), wopts=self._wopts)
            den, _ = quad(self._log_normal_pdf, a, b, args=(S0, t), wopts=self._wopts)
            
            results[boundary - S0] = num / den if den > 1e-10 else np.nan
        
        return pd.Series(results).sort_index().dropna()

    def _implied_S0(self, initial_delta: float) -> float:
        """
        Calculate the implied initial stock price S0 from a given initial delta.
        Uses:
            S0 = K * exp(d1 * sigma * sqrt(T) - (r + 0.5 * sigma^2) * T)
        with d1 = norm.ppf(initial_delta).
        """
        d1_value = norm.ppf(initial_delta)
        return self.K * np.exp(d1_value * self.sigma * np.sqrt(self.T) - (self.r + 0.5 * self.sigma**2) * self.T)

    def simulate_multiple(self, initial_deltas: list, sigma_moves: np.ndarray, delta_t: float) -> pd.DataFrame:
        """
        For each initial delta value, compute the expected delta changes and then reinterpolate the results
        onto a common grid. Returns a DataFrame with one column per initial delta.
        """
        sim_results = {}
        for delta in initial_deltas:
            S0 = self._implied_S0(delta)
            series = self.expected_delta_changes(S0, sigma_moves, delta_t)
            sim_results[f"Δ₀={delta} (S₀={S0:.2f})"] = series
        
        # Define a common grid across all simulation indexes.
        all_indices = np.concatenate([s.index.values for s in sim_results.values()])
        global_min = all_indices.min()
        global_max = all_indices.max()
        common_grid = np.linspace(global_min, global_max, 200)
        
        # Interpolate each simulation series onto the common grid.
        interpolated = {}
        for key, series in sim_results.items():
            series = series.sort_index()
            new_values = np.interp(common_grid, series.index.values, series.values)
            interpolated[key] = pd.Series(new_values, index=common_grid, name=key)
        
        return pd.DataFrame(interpolated)

    # -- New Methods for Option Charm --

    def initial_charm_change(self, initial_delta: float, delta_t: float) -> float:
        """
        For a given initial delta, compute the one-day change in delta due solely to option charm.
        Using the corresponding S0, evaluate:
            Δ_charm = charm(S0, T - delta_t) * delta_t.
        Returns a scalar value.
        """
        S0 = self._implied_S0(initial_delta)
        return self.charm(S0, self.T - delta_t) * delta_t

    def plot_charm_across_initial_deltas(self, initial_deltas: list, delta_t: float) -> pd.Series:
        """
        For each initial delta in the list, compute the one-day (delta_t) change in delta due to charm,
        using the initial stock price associated with that delta. Then, plot the results as a bar chart.
        
        Returns a pandas Series where the index is the initial delta and the value is the charm-induced change.
        """
        charm_changes = {}
        for d in initial_deltas:
            charm_changes[d] = self.initial_charm_change(d, delta_t)
        # Plot using the default canvas.
        fig, ax = self.default_canvas
        ax.bar(charm_changes.keys(), charm_changes.values(), color='skyblue')
        ax.set_xlabel("Initial Delta")
        ax.set_ylabel("Charm-induced Δ Change (per day)")
        ax.set_title("Change in Option Delta due to Charm Across Initial Deltas")
        ax.grid(True, axis='y')
        plt.show()
        return pd.Series(charm_changes)

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # 7. Initial parameters:
    params = {
        'K': 100,            # Strike price
        'r': 0.05,           # Risk-free rate
        'sigma': 0.158,      # Annual volatility
        'T': 5.0 / 252,      # Time to expiration (~5 trading days)
        'mu': 0.07           # Real-world drift
    }
    
    calculator = DeltaCalculator(**params)
    sigma_moves = np.arange(-4, 4.1, 0.1)
    initial_deltas = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    delta_t = 1/252  # One-day time step
    
    # Compute and display the expected delta differences.
    df_delta = calculator.simulate_multiple(initial_deltas, sigma_moves, delta_t)
    print("Expected Delta Differences (Interpolated):")
    print(df_delta.head())
    
    # Plot expected delta differences using the default canvas.
    fig, ax = calculator.default_canvas
    df_delta.plot(ax=ax)
    ax.set_xlabel("Price Deviation from S₀")
    ax.set_ylabel("Expected Δ Difference")
    ax.set_title("Expected Delta Differences vs Price Deviation")
    ax.grid(True, axis='y')
    plt.show()
    
    # For each initial delta, output the one-day charm-induced change in delta.
    print("One-day Charm-induced Δ Change:")
    for d in initial_deltas:
        charm_change = calculator.initial_charm_change(d, delta_t)
        print(f"Initial delta {d:0.2f} => Charm Δ Change: {charm_change:.6f}")
    
    # Plot the charm-induced change across all initial deltas.
    df_charm = calculator.plot_charm_across_initial_deltas(initial_deltas, delta_t)
    print("Charm-induced Δ Change across Initial Deltas:")
    print(df_charm)
    
    # --- New Summary Section ---
    # Build a summary table comparing initial delta, initial gamma, and one-day charm-induced delta change.
    summary_data = []
    for d in initial_deltas:
        S0 = calculator._implied_S0(d)
        # Although d is our target delta, we recalc delta for verification:
        delta_val = calculator.delta(S0, calculator.T)
        gamma_val = calculator.gamma(S0, calculator.T)
        charm_val = calculator.initial_charm_change(d, delta_t)
        summary_data.append({
            "Initial Delta": d,
            "Implied S₀": S0,
            "Calculated Delta": delta_val,
            "Initial Gamma": gamma_val,
            "Initial Charm": charm_val
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nSummary of Initial Greeks and Charm Effect:")
    print(summary_df)
    
    # Plot the summary as a bar chart (comparing Delta, Gamma, and Charm).
    fig, ax = calculator.default_canvas
    # For clarity, we plot Gamma and Charm; note that Delta is essentially the input target.
    width = 0.25
    x = np.arange(len(summary_df))
    
    ax.bar(x - width, summary_df["Initial Delta"], width, label="Initial Delta", color='lightgreen')
    ax.bar(x, summary_df["Initial Gamma"], width, label="Initial Gamma", color='salmon')
    ax.bar(x + width, summary_df["Initial Charm"], width, label="Initial Charm", color='skyblue')
    
    ax.set_xlabel("Option (by Initial Delta)")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Initial Delta"].round(2))
    ax.set_ylabel("Value")
    ax.set_title("Summary of Initial Delta, Gamma, and One-Day Charm Effect")
    ax.legend()
    ax.grid(True, axis='y')
    plt.show()
