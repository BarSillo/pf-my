import pytest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from rlhedge.basis_functions import *

try:
    from rlhedge.discrete_bs import DiscreteBlackScholes
except ImportError:
    from discrete_bs import DiscreteBlackScholes

# Test data creation helper
def create_test_data(num_paths=10_000, num_steps=6):
    s0, strike, vol = 100.0, 100.0, 0.2
    T, r, mu = 1.0, 0.05, 0.1
    numSteps = num_steps
    dt = T / numSteps

    Z1 = np.random.normal(0,1,size=(num_paths, numSteps+1))
    sVals = np.zeros_like(Z1)
    sVals[:, 0] = s0
    
    for step in range(numSteps):
        sVals[:, step+1] = sVals[:, step] * np.exp(
            (mu - 1/2 * vol ** 2)*dt + (vol * np.sqrt(dt) * Z1[:, step+1])
        )

    delta_S = sVals[:, 1:] - np.exp(r * dt) * sVals[:, :numSteps]
    delta_S_hat = np.apply_along_axis(lambda x: x - np.mean(x), axis=0, arr=delta_S)

    X = - (mu - 0.5 * vol ** 2) * np.arange(numSteps + 1) * dt + np.log(sVals)
    return X

def test_create_test_data():
    X = create_test_data(num_paths=3, num_steps=2)
    assert isinstance(X, np.ndarray)
    assert X.shape == (3, 3)  # (num_paths, num_steps + 1)

@pytest.mark.parametrize("num_basis", [4])
def test_bspline_basis(num_basis):
    X = create_test_data(num_paths=2, num_steps=1)
    basis = BSplineBasis()
    data = basis.compute_basis(X, num_basis=num_basis)
    assert data.shape == (2, 2, num_basis)  # (steps+1, paths, basis)

def test_polynomial_basis():
    x = np.array([[2.0], [3.0]])
    basis = PolynomialBasis()
    data = basis.compute_basis(x, num_basis=3)
    
    # Test bias term
    assert np.allclose(data[0, :, 0], 1.0)
    # Test linear term
    assert np.allclose(data[0, :, 1], [2.0, 3.0])
    # Test quadratic term
    assert np.allclose(data[0, :, 2], [4.0, 9.0])

def test_sklearn_basis():
    x = np.array([[2.0], [3.0]])
    basis = SKLearnPolynomialBasis()
    data = basis.compute_basis(x, num_basis=3)
    
    # Test bias term
    assert np.allclose(data[0, :, 0], 1.0)
    # Test linear term
    assert np.allclose(data[0, :, 1], [2.0, 3.0])
    # Test quadratic term
    assert np.allclose(data[0, :, 2], [4.0, 9.0])

def test_spline_transformer():
    X = create_test_data(num_paths=2, num_steps=1)
    basis = SKLearnSplineTransformer(degree=3, n_knots=4)
    data = basis.compute_basis(X, num_basis=6)
    
    # Test shape
    assert data.shape == (2, 2, 6)  # (steps+1, paths, basis)
    # Test for finite values
    assert np.all(np.isfinite(data))

if __name__ == '__main__':
    pytest.main([__file__])
