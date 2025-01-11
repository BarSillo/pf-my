# **Mathematical Formulation of Realized Delta**

The realized delta is computed as the expected value of the Black-Scholes delta across different stock prices, weighted by the probability density of those stock prices. Mathematically, the realized delta can be expressed as:

\[
\text{Realized Delta} = \frac{1}{3\sigma} \int_{S_{\text{low}}}^{S_{\text{high}}} \Delta(S') f(S') dS'
\]

## **Explanation of Terms**

### 1. **Black-Scholes Delta at Stock Price** \( S' \):

\[
\Delta(S') = N(d_1(S'))
\]
where

\[
d_1(S') = \frac{\ln\left( \frac{S'}{K} \right) + \left( r + \frac{1}{2}\sigma^2 \right)T}{\sigma \sqrt{T}}
\]

Here:

- \( K = S \cdot e^{-\text{log-moneyness}} \) is the rescaled strike price.
- \( r \) is the risk-free rate.
- \( \sigma \) is the volatility.
- \( T \) is the time to maturity.
- \( N(\cdot) \) is the cumulative distribution function of the standard normal distribution.

---

### 2. **Log-Normal Probability Density Function of Stock Price** \( S' \):

\[
f(S') = \frac{1}{S' \sigma \sqrt{2 \pi T}} \exp\left( -\frac{\left( \ln\left( \frac{S'}{S} \right) - \left( r - \frac{1}{2}\sigma^2 \right)T \right)^2}{2 \sigma^2 T} \right)
\]

---

### 3. **Bounds of Integration**:

\[
S_{\text{low}} = S \cdot e^{-3\sigma}, \quad S_{\text{high}} = S \cdot e^{3\sigma}
\]

---

### 4. **Normalization by \( 3\sigma \)**:

The integral is normalized by the size of the \( 3\sigma \) range to express the result in volatility units.
