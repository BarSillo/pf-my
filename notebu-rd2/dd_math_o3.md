# Normalizing the Change in Delta

One natural approach is to compare the computed change in delta to the first‐order Taylor expansion that uses the option’s gamma (the sensitivity of delta to price changes) and charm (the sensitivity of delta to the passage of time). For an option with initial delta $ \Delta(S_0,t_0) $, if the underlying moves by an amount $ dS = S_1-S_0 $ and time advances by $ dt = t_1-t_0 $, then a first‐order expansion gives

$$
\Delta(S_1,t_1) \approx \Delta(S_0,t_0) + \Gamma(S_0,t_0)\,dS + \chi(S_0,t_0)\,dt,
$$

where:
- $ \Gamma(S_0,t_0) $ is the option gamma,
- $ \chi(S_0,t_0) $ (often called “charm”) is the sensitivity of delta to time.

A natural normalized measure is then to define a ratio

$$
R = \frac{\Delta(S_1,t_1) - \Delta(S_0,t_0)}{\Gamma(S_0,t_0)\,dS + \chi(S_0,t_0)\,dt}\,.
$$

In this formulation, the numerator is the actual (or computed) change in delta, and the denominator is the “predicted” change based on the linear sensitivities. If the Taylor expansion holds perfectly, we would have $ R=1 $.

This ratio is dimensionless and has the advantage that it “normalizes” the change in delta by the option’s own sensitivities (gamma and charm). It allows you to compare the behavior across options that have different levels of gamma and charm. Alternatively, one might also report the relative error

$$
\text{Relative Error} = \frac{\Delta(S_1,t_1) - \Delta(S_0,t_0) - \left[\Gamma(S_0,t_0)\,dS + \chi(S_0,t_0)\,dt\right]}{\Gamma(S_0,t_0)\,dS + \chi(S_0,t_0)\,dt}\,,
$$

which would tell you the percentage by which the computed change deviates from the linear (first‐order) prediction.

In summary, by normalizing the computed change in delta as

$$
R = \frac{\Delta_{\text{computed}}}{\Gamma\,dS + \chi\,dt}\,,
$$

you obtain a dimensionless measure that can be compared across options with different gammas and charms. This approach focuses on the relative contributions of the underlying price move (weighted by gamma) and the time decay (weighted by charm) to the overall change in delta.
