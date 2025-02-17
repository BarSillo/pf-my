#!/usr/bin/env python
"""
dd-ds-quantlib-output.py

This script computes QuantLib outputs for all core_test_cases and prints them
in the format of core_test_cases.
"""

import QuantLib as ql
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from pprint import pprint

##############################################
# Helper function: get_quantlib_greeks
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
        'theta_daily': theta_daily,
        'vega': vega,
        'rho': rho
    }

##############################################
# Define core test cases (input parameters)
##############################################
core_cases = [
    {'case': 1, 'type': 'Call', 'S': 110, 'K': 100, 'T': 1,    'r': 4,  'sigma': 20},
    {'case': 2, 'type': 'Put',  'S': 100, 'K': 95,  'T': 0.25, 'r': 0,  'sigma': 40},
    {'case': 3, 'type': 'Call', 'S': 100, 'K': 100, 'T': 0.5,  'r': 5,  'sigma': 30},
    {'case': 4, 'type': 'Put',  'S': 95,  'K': 100, 'T': 1,    'r': 2,  'sigma': 25},
    {'case': 5, 'type': 'Call', 'S': 50,  'K': 55,  'T': 0.1,  'r': 1,  'sigma': 50}
]

##############################################
# Compute QuantLib outputs for each case
##############################################
output_cases = []

for case in core_cases:
    # Convert percent values to decimals for r and sigma
    r_decimal = case['r'] / 100.0
    sigma_decimal = case['sigma'] / 100.0
    T = case['T']
    S = case['S']
    K = case['K']
    option_type = ql.Option.Call if case['type'] == 'Call' else ql.Option.Put

    greeks = get_quantlib_greeks(option_type, S, K, T, r_decimal, sigma_decimal)

    # Round the output values for clarity
    rounded_greeks = { key: round(val, 5) for key, val in greeks.items() }

    # Following the original format, include "rho" only for Call cases 1 and 3.
    expected = {}
    expected['delta'] = rounded_greeks['delta']
    expected['gamma'] = rounded_greeks['gamma']
    expected['theta_daily'] = rounded_greeks['theta_daily']
    expected['vega'] = rounded_greeks['vega']
    if case['type'] == 'Call' and case['case'] in [1, 3]:
        expected['rho'] = rounded_greeks['rho']

    # Create a new dictionary in the same format as core_test_cases
    output_case = {
        'case': case['case'],
        'type': case['type'],
        'S': S,
        'K': K,
        'T': T,
        'r': case['r'],      # expressed as a percent
        'sigma': case['sigma'],  # expressed as a percent
        'expected': expected
    }
    output_cases.append(output_case)

##############################################
# Print the output in a pretty format.
##############################################
print("Updated core_test_cases based on QuantLib outputs:")
print("---------------------------------------------------")
pprint(output_cases)
