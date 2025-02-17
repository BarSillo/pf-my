# Theta Decay Formulation

This document provides the mathematical foundation for calculating theta decay in European call options, as implemented in the `OptionDeltaSimulator` class.

---

## Black-Scholes Theta Formula
For a European call option, the instantaneous theta (time decay) is given by:

$$
\Theta = -\frac{S \sigma \phi(d_1)}{2\sqrt{T}} - rKe^{-rT}N(d_2)
$$

Where:
- $ \phi(\cdot) $: Standard normal probability density function (PDF)
- $ N(\cdot) $: Standard normal cumulative distribution function (CDF)
- $$ d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}} $$
- $ d_2 = d_1 - \sigma\sqrt{T} $

---

## Variable Definitions
| Symbol | Description                        | Units         |
|--------|------------------------------------|---------------|
| \( S \) | Current stock price                | Dollar amount |
| \( K \) | Strike price                       | Dollar amount |
| \( r \) | Risk-free interest rate            | Annualized    |
| \( \sigma \) | Volatility                    | Annualized    |
| \( T \) | Time to expiration                 | Years         |
| \( t \) | Current time                       | Years         |

---

## Theta Decay Approximation
The dollar-value time decay over a small time interval $ \Delta t $ is approximated by:

$$
\Delta P_{\text{time}} \approx \Theta \cdot \Delta t
$$

Where:
- \( \Theta \): Instantaneous theta (per year)
- \( \Delta t \): Time step (years)

---

## Key Properties
1. **Sign Convention**:  
   Theta is typically negative for long option positions, representing time value erosion.

2. **Time Sensitivity**:  
   $$
   \frac{\partial \Theta}{\partial T} > 0 \quad \text{(Theta becomes more negative as expiration approaches)}
   $$

3. **Moneyness Relationship**:  
   - At-the-money (ATM) options have highest absolute theta  
   - Deep in/out-of-the-money options have lower theta

---

## Put Option Theta
For European put options, the theta formula becomes:

$$
\Theta_{\text{put}} = -\frac{S \sigma \phi(d_1)}{2\sqrt{T}} + rKe^{-rT}N(-d_2)
$$

---

## Time Decay Profile
Theta follows a characteristic curve versus time to expiration:

$$
\Theta(T) \propto -\frac{1}{\sqrt{T}}
$$

This leads to accelerated time decay in the final weeks before expiration.

---

## Example Values
For typical parameters:
- \( S = K = \$100 \)
- \( $\sigma$ = 30\% \)
- \( r = 5\% \)
- $ T = 30 \text{ days} $

$$
\Theta \approx -0.0523/\text{year} \quad \Rightarrow \quad \Delta P_{\text{daily}} \approx -\$0.0002
$$

---

## Implementation Notes
1. **Edge Handling**:  
$ \Theta \to 0 $ as $ T \to 0 $ (at expiration)

2. **Numerical Stability**:  
   Requires careful handling of near-expiration cases (\( T < 0.001 \) years)

3. **Dividend Adjustment**:  
   For dividend-paying stocks, modify \( d_1 \) with \( q \):
   $$
   d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}
   $$


## 1. Delta Evolution via Gamma

For small price moves, the change in delta can be approximated using gamma:

$$
\Delta_1 - \Delta_0 \approx \Gamma(S_0) \cdot (S_1 - S_0)
$$

Where:
- $\Gamma(S_0) = \frac{\phi(d_1(S_0))}{S_0 \sigma \sqrt{T - t}}$ is the gamma at initial time
- $\phi(\cdot)$ is the standard normal PDF
- $d_1(S) = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)(T-t)}{\sigma\sqrt{T-t}}$

For larger moves (e.g., multi-sigma moves), we need to integrate gamma along the price path:

$$
\Delta_1 - \Delta_0 = \int_{S_0}^{S_1} \Gamma(S') dS'
$$

This integral can be expressed as:

$$
\int_{S_0}^{S_1} \frac{\phi(d_1(S'))}{S' \sigma \sqrt{T - t}} dS'
$$

### Key Components:
1. **Gamma Expression**  
   $\Gamma(S) = \frac{\partial^2 C}{\partial S^2} = \frac{\phi(d_1(S))}{S \sigma \sqrt{T - t}}$

2. **Delta Difference**  
   $\mathbb{E}[\Delta_1 - \Delta_0] = \mathbb{E}\left[\int_{S_0}^{S_1} \Gamma(S') dS'\right]$

3. **Conditional Expectation**  
   For price moves between $S_{low}$ and $S_{high}$:
   
   $$
   \mathbb{E}[\Delta_1 - \Delta_0|S_1 \in (S_{low}, S_{high})] = \frac{\int_{S_{low}}^{S_{high}} \left(\int_{S_0}^{S} \Gamma(S') dS'\right) f(S) dS}{\int_{S_{low}}^{S_{high}} f(S) dS}
   $$

Where $f(S)$ is the risk-neutral probability density function of the stock price.

## 6. Recommended Formulation

For practical implementation of delta difference calculations, use this three-step approach:

### Step 1: Stock Price Probability Density
Compute the risk-neutral PDF of $S_1$:

$$
f(S_1) = \frac{1}{S_1 \sigma \sqrt{\Delta t}} \phi\left(\frac{\ln(S_1/S_0) - (\mu - 0.5\sigma^2)\Delta t}{\sigma \sqrt{\Delta t}}\right)
$$

Where:
- $\phi(\cdot)$ is the standard normal PDF
- $\mu$ is the real-world drift
- $\sigma$ is the annualized volatility

### Step 2: Bin Boundary Definition
Define price deviation bins using:

$$
S_{low}^k = S_0(1 - k\sigma_d), \quad S_{high}^k = S_0(1 - (k-1)\sigma_d) \quad \text{(negative moves)}
$$

$$
S_{low}^k = S_0(1 + (k-1)\sigma_d), \quad S_{high}^k = S_0(1 + k\sigma_d) \quad \text{(positive moves)}
$$

Where $\sigma_d = \sigma\sqrt{\Delta t}$ is the daily volatility.

### Step 3: Delta Difference Calculation
For each bin $[S_{low}, S_{high}]$, compute:

$$
\mathbb{E}[\Delta_1 - \Delta_0] = \frac{\int_{S_{low}}^{S_{high}} \underbrace{(N(d_1(S)) - \Delta_0}_{\text{Delta difference}} \cdot \overbrace{f(S)}^{\text{PDF}} dS}{\int_{S_{low}}^{S_{high}} f(S) dS}
$$

Where:
- $N(d_1(S)) = \Phi\left(\frac{\ln(S/K) + (r + 0.5\sigma^2)(T-\Delta t)}{\sigma\sqrt{T-\Delta t}}\right)$
- $T$ is the original time to maturity
- $r$ is the risk-free rate

### Implementation Notes:
1. Use numerical integration (e.g., Gauss-Legendre quadrature)
2. Cache frequent calculations (e.g., $\sigma\sqrt{\Delta t}$)
3. Handle edge cases where $f(S) \approx 0$ using log-probabilities