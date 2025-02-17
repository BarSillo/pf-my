import numpy as np
import time
import bspline
import bspline.splinelab as splinelab
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from abc import ABC, abstractmethod
import os
import sys
# from rlhedge import DiscreteBlackScholes

# Add parent directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Try package-style import first
    from rlhedge.custom_logging import setup_logger
except ImportError:
    # Fall back to direct import
    from rlhedge.custom_logging import setup_logger

logger = setup_logger(__name__)


class BasisFunction(ABC):
    """Abstract base class for basis function implementations"""
    def __init__(self, ncolloc=12, p=4):
        self.ncolloc = ncolloc
        self.num_basis = ncolloc
        self.p = p
    
    @abstractmethod
    def compute_basis(self, X):
        """Compute basis function expansion
        
        Args:
            X: Input data of shape (num_paths, num_steps + 1)
            
        Returns:
            data: Array of shape (num_steps + 1, num_paths, num_basis)
        """
        pass


class BSplineBasis(BasisFunction):
    """B-spline basis function implementation"""
    # inherit init from parent class:

    def __init__(self, p=4, ncolloc = 12):
        super().__init__(ncolloc, p)

    def compute_basis(self, X):
        num_steps = X.shape[1] - 1
        num_paths = X.shape[0]
        data = np.zeros((num_steps + 1, num_paths, self.ncolloc))
        
        # Create knots and basis
        X_min, X_max = np.min(X), np.max(X)
        tau = np.linspace(X_min, X_max, self.ncolloc)  # These are the sites to which we would like to interpolate
        k = splinelab.aptknt(tau, self.p)
        basis = bspline.Bspline(k, self.p)
        
        logger.info(f'Starting basis expansion with num_basis={self.ncolloc}')
        t_0 = time.time()
        
        for ix in np.arange(num_steps + 1):
            x = X[:, ix]
            data[ix, :, :] = np.array([basis(el) for el in x])
            if ix % 10 == 0:  # Log progress every 10 steps
                logger.info(f'Processed step {ix}/{num_steps}')
                
        t_end = time.time()
        
        logger.info(f'Basis expansion complete. Shape: {data.shape}')
        logger.info(f'Time Cost of basis expansion: {t_end - t_0:.3f} seconds')
        return data


class PolynomialBasis(BasisFunction):
    """Polynomial basis function implementation"""

    def __init__(self, p=4, ncolloc = 12):
        super().__init__(ncolloc, p)    
        self.num_basis = ncolloc
    
    def compute_basis(self, X):
        num_steps = X.shape[1] - 1
        num_paths = X.shape[0]
        data = np.zeros((num_steps + 1, num_paths, self.num_basis))
        
        logger.info(f'Starting polynomial basis expansion with num_basis={self.num_basis}')
        t_0 = time.time()
        
        for ix in np.arange(num_steps + 1):
            x = X[:, ix]
            # Compute polynomial basis up to degree num_basis-1
            for degree in range(self.num_basis):
                data[ix, :, degree] = x ** degree
            if ix % 10 == 0:  # Log progress every 10 steps
                logger.info(f'Processed step {ix}/{num_steps}')
                
        t_end = time.time()
        
        logger.info(f'Polynomial basis expansion complete. Shape: {data.shape}')
        logger.info(f'Time Cost of basis expansion: {t_end - t_0:.3f} seconds')
        return data


class SKLearnPolynomialBasis(BasisFunction):
    """Polynomial basis function implementation using sklearn's PolynomialFeatures"""
    
    def __init__(self, include_bias=True, p=4, ncolloc = 12):
        self.include_bias = include_bias
        super().__init__(ncolloc, p)  # Call the parent class constructor   
        
    def compute_basis(self, X, num_basis):
        num_steps = X.shape[1] - 1
        num_paths = X.shape[0]
        
        # Calculate degree needed to get num_basis features
        # For degree d: number of features = C(n+d,d) where n=1 (input dimension)
        # In our case: num_features = d + 1 when include_bias=True
        # So: degree = num_basis - 1
        degree = num_basis - 1 if self.include_bias else num_basis
        
        logger.info(f'Starting sklearn polynomial basis expansion with degree={degree}')
        t_0 = time.time()
        
        # Initialize transformer
        poly = PolynomialFeatures(degree=degree, include_bias=self.include_bias)

        # Pre-allocate output array
        data = np.zeros((num_steps + 1, num_paths, num_basis))
        
        for ix in np.arange(num_steps + 1):
            x = X[:, ix].reshape(-1, 1)  # reshape for sklearn
            data[ix] = poly.fit_transform(x)
            if ix % 10 == 0:  # Log progress every 10 steps
                    logger.info(f'Processed step {ix}/{num_steps}')
                
        t_end = time.time()
        
        logger.info(f'Sklearn polynomial basis expansion complete. Shape: {data.shape}')
        logger.info(f'Time Cost of basis expansion: {t_end - t_0:.3f} seconds')
        return data



class SKLearnSplineTransformer(BasisFunction):
    """Spline basis function implementation using sklearn's SplineTransformer"""
    #inherit init from parent class
    def __init__(self, p=4, ncolloc = 12, include_bias=True):
        super().__init__(ncolloc, p)  # Call the parent class constructor   
        self.degree = p
        self.n_knots = ncolloc
        self.include_bias = include_bias
        
    def compute_basis(self, X):
        num_steps = X.shape[1] - 1
        num_paths = X.shape[0]
        
        # Calculate n_knots if not specified
        if self.n_knots is None:
            # For degree d and n_knots k, number of basis functions = k + d - 1
            # So: k = num_basis - d + 1
            self.n_knots = max(2, num_basis - self.degree + 1)
        
        logger.info(f'Starting spline basis expansion with degree={self.degree}, n_knots={self.n_knots}')
        t_0 = time.time()
        
        # Initialize transformer
        spline = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=self.include_bias,
            extrapolation="linear"  # Use linear interpolation outside the domain
        )
        
        t_0 = time.time()
        num_basis = self.n_knots + self.degree - 1
        data = np.zeros((num_steps + 1, num_paths, num_basis))
        for ix in np.arange(num_steps + 1):
            x = X[:, ix].reshape(-1, 1)  # reshape for sklearn
            expanded = spline.fit_transform(x)
            assert expanded.shape[1] == num_basis, f"Number of basis functions {expanded.shape[1]} does not match n_knots + degree - 1"
            data[ix] = expanded

        # alternatively fit-transform once and reshape
        # data = spline.fit_transform(X)
        #reshape data to(num_paths, num_basis, num_steps + 1)
        # data = np.reshape(data, (num_steps + 1, num_paths, num_basis))
        t_end = time.time()
        
        logger.info(f'Spline basis expansion complete. Shape: {data.shape}')
        logger.info(f'Time Cost of basis expansion: {t_end - t_0:.3f} seconds')
        return data



    

    

  