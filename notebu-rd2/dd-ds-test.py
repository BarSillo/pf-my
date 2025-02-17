#!/usr/bin/env python
# dd-ds-test.py

"""
Execution Example:
pytest -q -k "Case_1" dd-ds-test.py
    -q: quiet mode
    -k: pattern match case

or

pytest dd-ds-test.py
"""

import pytest
import QuantLib as ql
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from dd_ds import BSOption  # Make sure BSOption is imported from your module

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPyPacked.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPyObject.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

##############################################
# Helper functions for QuantLib pricing
##############################################
def get_quantlib_greeks(option_type, S, K, T, r, sigma):
    """
    Build a European option in QuantLib and return its Greeks.
    
    Args:
        option_type: ql.Option.Call or ql.Option.Put
        S: underlying price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate (decimal)
        sigma: volatility (decimal)
    Returns:
        A dictionary of greeks: delta, gamma, theta (daily), vega, rho.
    """
    # Set evaluation date
    todays_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = todays_date
    # Maturity date: add T years (approximate using days)
    maturity_date = todays_date + int(round(T * 365))
    payoff = ql.PlainVanillaPayoff(option_type, K)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)

    # Build the Black-Scholes process.
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, r, ql.Actual365Fixed()))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, 0.0, ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(todays_date, ql.NullCalendar(), sigma, ql.Actual365Fixed())
    )
    underlying = ql.QuoteHandle(ql.SimpleQuote(S))
    process = ql.BlackScholesMertonProcess(underlying, dividend_ts, flat_ts, vol_ts)
    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)

    # Retrieve greeks.
    delta = option.delta()
    gamma = option.gamma()
    theta_annual = option.theta()  # annual theta
    theta_daily = theta_annual / 365.0  # convert to daily theta
    vega = option.vega()
    rho = option.rho()
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta_daily,
        'vega': vega,
        'rho': rho
    }

##############################################
# Core Test Case Matrix (generated from QuantLib)
##############################################
core_test_cases = [
    {'K': 100,
     'S': 110,
     'T': 1,
     'case': 1,
     'expected': {'delta': 0.78129,
                  'gamma': 0.01341,
                  'rho': 68.97301,
                  'theta_daily': -0.01645,
                  'vega': 32.46057},
     'r': 4,
     'sigma': 20,
     'type': 'Call'},
    {'K': 95,
     'S': 100,
     'T': 0.25,
     'case': 2,
     'expected': {'delta': -0.36067,
                  'gamma': 0.01874,
                  'theta_daily': -0.04108,
                  'vega': 18.69212},
     'r': 0,
     'sigma': 40,
     'type': 'Put'},
    {'K': 100,
     'S': 100,
     'T': 0.5,
     'case': 3,
     'expected': {'delta': 0.58847,
                  'gamma': 0.01837,
                  'rho': 24.54595,
                  'theta_daily': -0.02939,
                  'vega': 27.47525},
     'r': 5,
     'sigma': 30,
     'type': 'Call'},
    {'K': 100,
     'S': 95,
     'T': 1,
     'case': 4,
     'expected': {'delta': -0.50007,
                  'gamma': 0.0168,
                  'theta_daily': -0.00976,
                  'vega': 37.89952},
     'r': 2,
     'sigma': 25,
     'type': 'Put'},
    {'K': 55,
     'S': 50,
     'T': 0.1,
     'case': 5,
     'expected': {'delta': 0.30078,
                  'gamma': 0.04434,
                  'theta_daily': -0.03833,
                  'vega': 5.46609},
     'r': 1,
     'sigma': 50,
     'type': 'Call'}
]

# Tolerance standards
tol_delta = 2e-3
tol_gamma = 1e-4
tol_theta = 5e-3
tol_others = 1e-2  # relaxed for vega and rho

##############################################
# Volatility Surface Test Matrix
##############################################
vol_surface_cases = [
    {
        'sigma': 20,
        'call_delta': 0.637,
        'put_delta': -0.363,
        'gamma': 0.01877
    },
    {
        'sigma': 40,
        'call_delta': 0.628,
        'put_delta': -0.372,
        'gamma': 0.00946
    },
    {
        'sigma': 60,
        'call_delta': 0.649,
        'put_delta': -0.351,
        'gamma': 0.00618
    }
]

S_default = 100
K_default = 100
T_default = 1
r_default = 0.05

##############################################
# QuantLib Core Test Cases
##############################################
@pytest.mark.parametrize("case", core_test_cases, ids=lambda c: f"Case_{c['case']}")
def test_quantlib_core_greeks(case):
    # Convert percentages to decimals.
    r = case['r'] / 100.0
    sigma = case['sigma'] / 100.0
    T = case['T']
    S = case['S']
    K = case['K']
    option_type = ql.Option.Call if case['type'] == 'Call' else ql.Option.Put

    greeks = get_quantlib_greeks(option_type, S, K, T, r, sigma)

    # Compare QuantLib’s greeks to the updated expected values.
    assert abs(greeks['delta'] - case['expected']['delta']) < tol_delta, f"Delta mismatch in case {case['case']}"
    assert abs(greeks['gamma'] - case['expected']['gamma']) < tol_gamma, f"Gamma mismatch in case {case['case']}"
    assert abs(greeks['theta'] - case['expected']['theta_daily']) < tol_theta, f"Theta mismatch in case {case['case']}"
    assert abs(greeks['vega'] - case['expected']['vega']) < tol_others, f"Vega mismatch in case {case['case']}"
    if 'rho' in case['expected']:
        assert abs(greeks['rho'] - case['expected']['rho']) < tol_others, f"Rho mismatch in case {case['case']}"

##############################################
# QuantLib Volatility Surface Test Cases
##############################################
@pytest.mark.parametrize("case", vol_surface_cases, ids=lambda c: f"Vol_{c['sigma']}")
def test_quantlib_vol_surface(case):
    sigma = case['sigma'] / 100.0
    option_type_call = ql.Option.Call
    option_type_put = ql.Option.Put

    greeks_call = get_quantlib_greeks(option_type_call, S_default, K_default, T_default, r_default, sigma)
    greeks_put = get_quantlib_greeks(option_type_put, S_default, K_default, T_default, r_default, sigma)

    assert abs(greeks_call['delta'] - case['call_delta']) < tol_delta, f"Call Delta mismatch for sigma {case['sigma']}"
    assert abs(greeks_put['delta'] - case['put_delta']) < tol_delta, f"Put Delta mismatch for sigma {case['sigma']}"
    assert abs(greeks_call['gamma'] - case['gamma']) < tol_gamma, f"Gamma mismatch for sigma {case['sigma']}"

##############################################
# BSOption Class Tests Using Finite Difference Checks
##############################################
# Helper function for Black-Scholes call price.
def bs_call_price(S, bs_option: BSOption):
    # Black-Scholes call price: S*N(d1) - K*exp(-r*t)*N(d2)
    t = bs_option.T
    d1 = bs_option._d1(S, t)
    d2 = d1 - bs_option.sigma * sqrt(t)
    price = S * norm.cdf(d1) - bs_option.K * exp(-bs_option.r * t) * norm.cdf(d2)
    return price

def numerical_delta(S, bs_option: BSOption, epsilon=1e-4):
    price_plus = bs_call_price(S + epsilon, bs_option)
    price_minus = bs_call_price(S - epsilon, bs_option)
    return (price_plus - price_minus) / (2 * epsilon)

def numerical_theta(S, bs_option: BSOption, dt=1/365):
    """
    Compute daily theta.
    Since dt is one day expressed in years (1/365), the daily theta is just the difference
    in price over one day.
    """
    price_today = bs_call_price(S, bs_option)
    if bs_option.T - dt <= 0:
        return 0.0
    bs_option_new = BSOption(bs_option.K, bs_option.r, bs_option.sigma, bs_option.T - dt)
    price_tomorrow = bs_call_price(S, bs_option_new)
    return price_tomorrow - price_today

# Test BSOption for call cases via finite difference checks.
@pytest.mark.parametrize("case", [c for c in core_test_cases if c['type'] == 'Call'],
                         ids=lambda c: f"BSOption_Case_{c['case']}")
def test_bsoption_delta_and_theta(case):
    r = case['r'] / 100.0
    sigma = case['sigma'] / 100.0
    T = case['T']
    K = case['K']
    S = case['S']

    bs_option = BSOption(K, r, sigma, T)

    # Analytical delta from BSOption (call delta)
    analytical_delta = bs_option.delta(S, T)
    # Numerical delta computed by finite differences
    num_delta = numerical_delta(S, bs_option)
    assert abs(analytical_delta - num_delta) < tol_delta, f"BSOption Delta numerical check failed for case {case['case']}"

    # Convert BSOption.theta (annualized) to daily theta.
    analytical_theta_daily = bs_option.theta(S, T) / 365.0
    num_theta_daily = numerical_theta(S, bs_option)
    assert abs(analytical_theta_daily - num_theta_daily) < tol_theta, f"BSOption Theta numerical check failed for case {case['case']}"

# Simple test to verify that BSOption.charm returns a float.
@pytest.mark.parametrize("case", [c for c in core_test_cases if c['type'] == 'Call'],
                         ids=lambda c: f"BSOption_Charm_Case_{c['case']}")
def test_bsoption_charm(case):
    r = case['r'] / 100.0
    sigma = case['sigma'] / 100.0
    T = case['T']
    K = case['K']
    S = case['S']

    bs_option = BSOption(K, r, sigma, T)
    charm_value = bs_option.charm(S, T)
    assert isinstance(charm_value, float), f"BSOption charm did not return a float for case {case['case']}"

##############################################
# NEW: BSOption Direct Comparison to Expected Core and Vol Surface Cases
##############################################
# For call cases only (since BSOption implements call pricing/greeks).

@pytest.mark.parametrize("case", [c for c in core_test_cases if c['type'] == 'Call'],
                         ids=lambda c: f"BSOption_Core_{c['case']}")
def test_bsoption_core_greeks(case):
    r = case['r'] / 100.0
    sigma = case['sigma'] / 100.0
    T = case['T']
    K = case['K']
    S = case['S']

    bs_option = BSOption(K, r, sigma, T)
    
    # Test delta
    computed_delta = bs_option.delta(S, T)
    expected_delta = case['expected']['delta']
    assert abs(computed_delta - expected_delta) < tol_delta, (
        f"BSOption delta mismatch for core case {case['case']}: computed {computed_delta}, expected {expected_delta}"
    )
    
    # Test theta (daily)
    computed_theta_daily = bs_option.theta(S, T) / 365.0
    expected_theta_daily = case['expected']['theta_daily']
    assert abs(computed_theta_daily - expected_theta_daily) < tol_theta, (
        f"BSOption theta mismatch for core case {case['case']}: computed {computed_theta_daily}, expected {expected_theta_daily}"
    )

@pytest.mark.parametrize("case", vol_surface_cases, ids=lambda c: f"BSOption_Vol_{c['sigma']}")
def test_bsoption_vol_surface_call_delta(case):
    sigma = case['sigma'] / 100.0
    S = S_default
    K = K_default
    T = T_default
    r = r_default
    bs_option = BSOption(K, r, sigma, T)
    
    computed_delta = bs_option.delta(S, T)
    expected_delta = case['call_delta']
    assert abs(computed_delta - expected_delta) < tol_delta, (
        f"BSOption call delta mismatch for vol case sigma {case['sigma']}: computed {computed_delta}, expected {expected_delta}"
    )
