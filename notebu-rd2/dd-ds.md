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
- \( T = 30 \text{ days} \)

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