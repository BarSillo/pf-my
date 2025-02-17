import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt

class BSOption:
    """Black-Scholes option calculations with Greeks."""
    def __init__(self, K: float, r: float, sigma: float, T: float):
        """
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
        return (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * t) / (self.sigma * np.sqrt(t))

    def delta(self, S: float, t: float) -> float:
        """Call option delta."""
        return norm.cdf(self._d1(S, t))

    def gamma(self, S: float, t: float) -> float:
        d1 = self._d1(S, t)
        return norm.pdf(d1) / (S * self.sigma * np.sqrt(t))

class DeltaCalculator(BSOption):
    """Numerical integration–based delta change calculator."""
    def __init__(self, K: float, r: float, sigma: float, T: float, mu: float = 0.07):
        """
        Args:
            mu: Real-world drift parameter.
        """
        super().__init__(K, r, sigma, T)
        self.mu = mu
        self._wopts = None  # Optional: parameters for quad optimization

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
        For a given starting price S0 and a range of sigma moves, compute the
        expected delta change conditioned on the move reaching the boundary:
            S_b = S0 * (1 + sigma_move * sigma*sqrt(delta_t))
        
        Args:
            S0: Initial stock price.
            sigma_moves: Array of sigma multiples.
            delta_t: Time step for the move.
        
        Returns:
            pd.Series indexed by (S_b - S0) (i.e. price deviation) with the computed
            expected delta difference.
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
        
        Uses the rearranged Black-Scholes formula:
            S0 = K * exp(d1 * sigma*sqrt(T) - (r + 0.5*sigma^2)*T)
        where d1 = norm.ppf(initial_delta)
        """
        d1_value = norm.ppf(initial_delta)
        return self.K * np.exp(d1_value * self.sigma * np.sqrt(self.T) - (self.r + 0.5 * self.sigma**2) * self.T)

    def simulate_multiple(self, initial_deltas: list, sigma_moves: np.ndarray, delta_t: float) -> pd.DataFrame:
        """
        For each initial delta value, compute the expected delta changes and then
        reinterpolate the results onto a common grid. The output is a DataFrame
        with one column per initial delta.
        
        Args:
            initial_deltas: List of target initial deltas.
            sigma_moves: Array of sigma multiples.
            delta_t: Time step for the move.
            
        Returns:
            A pandas DataFrame whose columns correspond to different initial delta settings.
            The index is the common grid of price deviations (S_b - S0).
        """
        sim_results = {}
        for delta in initial_deltas:
            S0 = self._implied_S0(delta)
            series = self.expected_delta_changes(S0, sigma_moves, delta_t)
            sim_results[f"Δ₀={delta} (S₀={S0:.2f})"] = series
        
        # Because S0 differs across simulations, the price-deviation indexes may not line up.
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
        
        df = pd.DataFrame(interpolated)
        return df

# Example Usage
if __name__ == "__main__":
    # Set model parameters.
    params = {
        'K': 100,       # Strike price
        'r': 0.05,      # Risk-free rate
        'sigma': 0.2,   # Annual volatility
        'T': 1.0,       # Time to expiration
        'mu': 0.07      # Real-world drift
    }
    
    calculator = DeltaCalculator(**params)
    sigma_moves = np.arange(-6, 6.1, 0.1)
    initial_deltas = [0.4, 0.5, 0.6]
    delta_t = 1/252  # Daily time step
    
    # Compute the expected delta differences for each initial delta and combine into a DataFrame.
    df = calculator.simulate_multiple(initial_deltas, sigma_moves, delta_t)
    
    print("Expected Delta Differences (Interpolated):")
    print(df.head())
    
    # Plotting with custom settings:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#F5F5DC')  # Light beige for the figure background
    ax.set_facecolor('#FAF0E6')         # Lighter beige for the plot area
    
    df.plot(ax=ax)
    ax.set_xlabel("Price Deviation from S₀")
    ax.set_ylabel("Expected Δ Difference")
    ax.set_title("Expected Delta Differences vs Price Deviation")
    # Only display horizontal grid lines:
    ax.grid(True, axis='y')
    plt.show()
