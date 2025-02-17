# Black-Scholes Greeks Validation Test Plan

## Core Test Case Matrix

```math
\begin{array}{|c|c|c|c|c|c|c|c|l|}
\hline
\text{Case} & \text{Type} & S & K & T\,(\text{yrs}) & r\,(\%) & \sigma\,(\%) & \text{Expected Greeks} & \text{Source} \\
\hline
1 & \text{Call} & 110 & 100 & 1 & 4 & 20 & 
\begin{aligned}
\Delta &= 0.84 \\ 
\Gamma &= 0.019 \\
\Theta_{\text{daily}} &= -0.01 \\
\nu &= 0.25 \\
\rho &= 0.33
\end{aligned} & \text{[5]} \\
\hline
2 & \text{Put} & 100 & 95 & 0.25 & 0 & 40 & 
\begin{aligned}
\Delta &= -0.36 \\
\Gamma &= 0.019 \\
\Theta_{\text{daily}} &= -0.059 \\
\nu &= 0.38
\end{aligned} & \text{[4]} \\
\hline
3 & \text{Call} & 100 & 100 & 0.5 & 5 & 30 & 
\begin{aligned}
\Delta &= 0.62 \\
\Gamma &= 0.025 \\
\Theta_{\text{daily}} &= -0.03 \\
\nu &= 0.40
\end{aligned} & \text{[1]} \\
\hline
4 & \text{Put} & 95 & 100 & 1 & 2 & 25 & 
\begin{aligned}
\Delta &= -0.41 \\
\Gamma &= 0.022 \\
\Theta_{\text{daily}} &= -0.02 \\
\nu &= 0.35
\end{aligned} & \text{[6]} \\
\hline
5 & \text{Call} & 50 & 55 & 0.1 & 1 & 50 & 
\begin{aligned}
\Delta &= 0.32 \\
\Gamma &= 0.045 \\
\Theta_{\text{daily}} &= -0.08 \\
\nu &= 0.12
\end{aligned} & \text{[3]} \\
\hline
\end{array}
```

---

## Volatility Surface Test Matrix

```math
\begin{array}{|c|c|c|c|}
\hline
\sigma\,(\%) & \text{Call}\,\Delta & \text{Put}\,\Delta & \Gamma \\
\hline
20 & 0.64 & -0.36 & 0.025 \\
40 & 0.58 & -0.42 & 0.032 \\
60 & 0.53 & -0.47 & 0.028 \\
\hline
\end{array}
```

---

## Greek Symbols Legend
- $\Delta$: Delta
- $\Gamma$: Gamma  
- $\Theta$: Theta
- $\nu$: Vega
- $\rho$: Rho

## Validation Formulas
For numerical derivative checks:
```math
\Delta_{\text{num}} = \frac{C(S+\epsilon) - C(S-\epsilon)}{2\epsilon}
```

Put-Call Parity:
```math
C - P = S - Ke^{-rT}
```

## Tolerance Standards
```math
\begin{aligned}
|\Delta_{\text{analytical}} - \Delta_{\text{num}}| &< 10^{-3} \\
|\Gamma_{\text{analytical}} - \Gamma_{\text{num}}| &< 10^{-4} \\
|\Theta_{\text{daily}} - \Theta_{\text{num}}| &< 5\times10^{-3}
\end{aligned}
```

## References
1. Black-Scholes original paper  
5. Hull's Options Futures & Derivatives  
4. Wilmott Quantitative Finance  
3. Haug's Complete Guide  
6. Natenberg Option Volatility