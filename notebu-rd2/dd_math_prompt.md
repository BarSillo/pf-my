# Problem: Evaluate Delta Changes for Large Underlying Moves

## Objective
Develop Python code to mathematically evaluate the difference in option delta ($\Delta$) for large moves in the underlying price using numerical integration rather than Monte Carlo methods.

## Key Requirements
1. **Mathematical Foundation**:
   - Black-Scholes delta calculations
   - Gamma ($\Gamma$) integration approach:
     $$
     \Delta_1 - \Delta_0 = \int_{S_0}^{S_1} \Gamma(S) dS
     $$
   - Conditional expectation formulation:
     $$
     \mathbb{E}[\Delta_1 - \Delta_0 | S \in (a,b)] = \frac{\int_a^b (\Delta(S) - \Delta_0)f(S)dS}{\int_a^b f(S)dS}
     $$

2. **Python Implementation**:
   ```python
   class BSOption:
       """Base Black-Scholes calculations"""
       def delta(self, S: float, t: float) -> float: ...
       def gamma(self, S: float, t: float) -> float: ...

   class DeltaCalculator(BSOption):
       """Numerical integration implementation"""
       def expected_delta_changes(self, S0: float, sigma_moves: np.ndarray, delta_t: float) -> pd.Series: 
           # Uses scipy.integrate.quad with optimized bounds handling
   ```

3. **Key Features**:
   - Handle moves up to ±6σ
   - Integration from $S_0$ to move boundaries
   - Log-normal PDF for underlying price evolution
   - Optimized consecutive quad calls

4. **Expected Output**:
   - Pandas Series of expected delta changes
   - Indexed by price deviation from $S_0$
   - Visualization-ready format

## To Continue Development
1. **Next Steps**:
   ```python
   # Example restart point
   calculator = DeltaCalculator(K=100, r=0.05, sigma=0.2, T=1)
   results = calculator.expected_delta_changes(S0=100, sigma_moves=np.arange(-6,6.1,0.1), delta_t=1/252)
   ```

2. **Pending Optimizations**:
   - Cython integration for quad speedup
   - Adaptive subdivision reuse between calls
   - Batch interval processing

3. **Key Considerations**:
   - Numerical stability at extreme σ values
   - Real-world drift ($\mu$) vs risk-neutral measure
   - Time decay effects on gamma
   ```

## Full Context Snapshot
[Attach latest code implementation and mathematical formulation from previous messages]


More detailed Prompt:
======================
# Black-Scholes Delta Change Evaluation Package

## 1. Core Mathematical Formulation

### Delta-Gamma Relationship
For a move from $S_0$ to $S_1$:
$$ \Delta_1 - \Delta_0 = \int_{S_0}^{S_1} \Gamma(S) dS $$

### Conditional Expectation
For moves to boundary $S_b$:
$$ \mathbb{E}[\Delta_1 - \Delta_0 | S \in (S_0,S_b)] = \frac{\int_{S_0}^{S_b} (\Delta(S) - \Delta_0)f(S)dS}{\int_{S_0}^{S_b} f(S)dS} $$

### Log-Normal PDF
$$ f(S) = \frac{1}{S\sigma\sqrt{\Delta t}} \phi\left(\frac{\ln(S/S_0) - (\mu - 0.5\sigma^2)\Delta t}{\sigma\sqrt{\Delta t}}\right) $$

## 2. Complete Python Implementation

```python
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad

class BSOption:
    """Black-Scholes option calculations with Greeks"""
    def __init__(self, K: float, r: float, sigma: float, T: float):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    def _d1(self, S: float, t: float) -> float:
        return (np.log(S/self.K) + (self.r + 0.5*self.sigma**2)*t) / (self.sigma*np.sqrt(t))

    def delta(self, S: float, t: float) -> float:
        return norm.cdf(self._d1(S, t))

    def gamma(self, S: float, t: float) -> float:
        d1 = self._d1(S, t)
        return norm.pdf(d1) / (S * self.sigma * np.sqrt(t))

class DeltaCalculator(BSOption):
    """Numerical integration-based delta change calculator"""
    def __init__(self, K: float, r: float, sigma: float, T: float, mu: float = 0.07):
        super().__init__(K, r, sigma, T)
        self.mu = mu
        self._wopts = None  # For quad optimization

    def _log_normal_pdf(self, S: float, S0: float, t: float) -> float:
        drift = (self.mu - 0.5*self.sigma**2) * t
        scale = self.sigma * np.sqrt(t)
        z = (np.log(S/S0) - drift) / scale
        return norm.pdf(z) / (S * scale)

    def _delta_diff_integrand(self, S: float, S0: float, t: float) -> float:
        return (self.delta(S, self.T - t) - self.delta(S0, self.T)) * self._log_normal_pdf(S, S0, t)

    def expected_delta_changes(self, S0: float, sigma_moves: np.ndarray, delta_t: float) -> pd.Series:
        results = {}
        t = delta_t
        sigma_d = self.sigma * np.sqrt(t)
        
        for sigma_move in sigma_moves:
            boundary = S0 * (1 + sigma_move * sigma_d)
            a, b = sorted([S0, boundary])
            
            # Optimized quad calls with subdivision reuse
            num, _ = quad(self._delta_diff_integrand, a, b, args=(S0, t), 
                        wopts=self._wopts)
            den, _ = quad(self._log_normal_pdf, a, b, args=(S0, t),
                        wopts=self._wopts)
            
            results[boundary - S0] = num / den if den > 1e-10 else np.nan
        
        return pd.Series(results).sort_index().dropna()

# Example Usage
if __name__ == "__main__":
    params = {
        'K': 100,        # Strike
        'r': 0.05,       # Risk-free rate
        'sigma': 0.158,    # Volatility
        'T': 5.0 / 252,        # Time to expiration
        'mu': 0.07       # Real-world drift
    }
    
    calculator = DeltaCalculator(**params)
    sigma_moves = np.arange(-6, 6.1, 0.1)
    results = calculator.expected_delta_changes(
        S0=100, 
        sigma_moves=sigma_moves,
        delta_t=1/252
    )
```

## 3. Key Optimizations Implemented

1. **Quad Subdivision Reuse**  
   ```python
   wopts=self._wopts  # Stores adaptive integration parameters
   ```
   
2. **Vectorized Boundary Handling**  
   ```python
   a, b = sorted([S0, boundary])  # Auto-detect direction
   ```

3. **Numerical Stability Guards**  
   ```python
   den > 1e-10  # Skip ill-conditioned integrals
   ```

## 4. Pending Optimizations (For Future Work)

```python
# Cython Integrand (save as fast_integrand.pyx)
cdef public double integrand(int n, double *args) nogil:
    # Implement _delta_diff_integrand here
    # See scipy.integrate.LowLevelCallable docs
```

## 5. Key Parameters Table

| Parameter | Symbol | Example Value | Description |
|-----------|--------|---------------|-------------|
| K         | $K$    | 100           | Strike price |
| sigma     | $\sigma$ | 0.158       | Annual volatility |
| mu        | $\mu$  | 0.07          | Real-world drift |
| delta_t   | $\Delta t$ | 1/252     | Daily time step |

## 6. Expected Output Format

```python
# pd.Series with price deviations as index
-5.960594   -0.002142
-5.861283   -0.001873
... 
5.861283     0.002098
5.960594     0.001846
dtype: float64
```
## 7. initial parameters:
```py
    # Set model parameters.
    params = {
        'K': 100,       # Strike price
        'r': 0.05,      # Risk-free rate
        'sigma': 0.158,   # Annual volatility
        'T': 5.0 / 252,       # Time to expiration
        'mu': 0.07      # Real-world drift
    }
    
    calculator = DeltaCalculator(**params)
    sigma_moves = np.arange(-4, 4.1, 0.1)
    initial_deltas = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    delta_t = 1/252  # Daily time step
```

## 8. Plotting paramters:
Save these parameters as default canvas for plotting as a property of the class and use them in all plotting methods for the dataframe:
```python
    # Plotting with custom settings:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#F5F5DC')  # Light beige for the figure background
    ax.set_facecolor('#FAF0E6')         # Lighter beige for the plot area
```

## Restart Checklist
1. Verify Black-Scholes formulas match use case
2. Validate PDF implementation matches market data
3. Adjust sigma_moves range as needed
4. Implement pending Cython optimizations for production use
5. Change code to return a dataframe that has expected delta change for each value of underlying change. 
The dataframe should be indexed by change in underlying from initial value of stock.
6. Implement a method that would output change in inital delta of the option due to options charm and also have a method to plot the series of charm for each initial delta. For plotting charm use initial stock price initial associated with given initial delta. I want to see a single expected change in options delta due ta passage of delta_t which is initialized to be one day. I want to compare the effect of it across options that have diffrent initial deltas.
