"""
RLHedge - A reinforcement learning package for financial hedging
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .discrete_bs import DiscreteBlackScholes
from .basis_functions import (
    BasisFunction,
    SKLearnPolynomialBasis,
    SKLearnSplineTransformer,
)

__all__ = [
    "DiscreteBlackScholes",
    "BasisFunction",
    "SKLearnPolynomialBasis",
    "SKLearnSplineTransformer",
]
