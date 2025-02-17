from scipy.integrate import quad
from scipy.stats import lognorm, norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dd_ds import BSOption


class DeltaCalculator(BSOption):
    """Calculates delta changes with integration from S0 to move boundaries"""
    def __init__(self, K: float, r: float, sigma: float, T: float, mu: float = 0.07):
        super().__init__(K, r, sigma, T)
        self.mu = mu

    def _log_normal_pdf(self, S: float, S0: float, t: float) -> float:
        """Log-normal PDF of stock price moves"""
        drift = (self.mu - 0.5*self.sigma**2) * t
        scale = self.sigma * np.sqrt(t)
        z = (np.log(S/S0) - drift) / scale
        return norm.pdf(z) / (S * scale)

    def _delta_diff_integrand(self, S: float, S0: float, t: float) -> float:
        """Integrand for conditional delta difference"""
        return (self.delta(S, self.T - t) - self.delta(S0, self.T)) * self._log_normal_pdf(S, S0, t)

    def expected_delta_changes(self, S0: float, sigma_moves: np.ndarray, delta_t: float) -> pd.Series:
        """
        Calculate expected ΔΔ for moves from S0 to each boundary
        
        Args:
            S0: Initial stock price
            sigma_moves: Array of sigma boundaries (e.g., [-6, -5.9, ..., 5.9, 6])
            delta_t: Time step in years
            
        Returns:
            Series indexed by price deviation from S0
        """
        results = {}
        t = delta_t
        sigma_d = self.sigma * np.sqrt(t)
        
        for sigma_move in sigma_moves:
            # Convert sigma move to price boundary
            price_deviation = sigma_move * sigma_d * S0
            boundary = S0 + price_deviation
            
            # Set integration bounds
            if boundary > S0:
                a, b = S0, boundary
            elif boundary < S0:
                a, b = boundary, S0
            else:
                continue  # Skip zero move

            try:
                # Integrate numerator and denominator
                num, _ = quad(self._delta_diff_integrand, a, b, args=(S0, t))
                den, _ = quad(self._log_normal_pdf, a, b, args=(S0, t))
                
                if den > 1e-10:
                    results[price_deviation] = num / den
                else:
                    results[price_deviation] = np.nan
                    
            except Exception as e:
                print(f"Integration failed for {sigma_move}σ: {str(e)}")
                results[price_deviation] = np.nan

        return pd.Series(results).sort_index().dropna()
    

import numpy as np
from scipy.integrate import quad
from scipy import LowLevelCallable




# Python wrapper
class AdaptiveIntegrator:
    def __init__(self, a_param, b_param):
        self.wopts = None
        self.last_bounds = None
        self.total = 0.0
        self.args = (a_param, b_param)
        
        # Build LowLevelCallable
        from fast_integrand import integrand
        self.llc = LowLevelCallable.from_cython(
            "fast_integrand", "integrand",
            args=self.args
        )
    
    def integrate_step(self, new_lower, new_upper):
        if self.last_bounds is None:
            # Initial integration
            res, err, info = quad(
                self.llc, new_lower, new_upper,
                args=self.args, full_output=True,
                wopts=self.wopts
            )
        else:
            # Reuse previous subdivisions
            prev_lower, prev_upper = self.last_bounds
            if new_lower == prev_upper:  # Adjacent interval
                res, err, info = quad(
                    self.llc, new_lower, new_upper,
                    points=[prev_upper],  # Force subdivision reuse
                    wopts=self.wopts
                )
            else:
                res, err, info = quad(
                    self.llc, new_lower, new_upper,
                    wopts=self.wopts
                )
        
        # Store adaptive parameters for next call
        self.wopts = (info['momcom'], info['chebmo'])
        self.last_bounds = (new_lower, new_upper)
        self.total += res
        return res, self.total


def run_fast():
    # Usage Example
    integrator = AdaptiveIntegrator(a_param=0.5, b_param=2.0)

    # Consecutive calls with moving window
    bounds_sequence = [(0, 1), (1, 1.5), (1.5, 2.0)]
    for lower, upper in bounds_sequence:
        step_result, total = integrator.integrate_step(lower, upper)
        print(f"Interval ({lower}-{upper}): {step_result:.4f}, Total: {total:.4f}")


# Example usage
if __name__ == "__main__":
    # Initialize parameters
    K = 100       # Strike price
    r = 0.05      # Risk-free rate
    sigma = 0.158  # Annual volatility
    T = 1.0       # Time to expiration (years)
    mu = 0.07     # Real-world drift
    S0 = 100      # Initial stock price
    delta_t = 1/252  # Daily time step
    
    # Create calculator
    calculator = DeltaCalculator(K=K, r=r, sigma=sigma, T=T, mu=mu)
    
    # Generate sigma moves from -6 to +6 in 0.1 steps
    sigma_steps = np.arange(-6, 6.1, 0.1)
    
    # Calculate expected delta changes
    results = calculator.expected_delta_changes(S0, sigma_steps, delta_t)
    
    # Plot results
    results.plot(style='o-', markersize=3)
    plt.title('Expected Delta Changes for Large Moves')
    plt.xlabel('Price Deviation from S0')
    plt.ylabel('Expected Delta Change')
    plt.grid(True)
    plt.show()