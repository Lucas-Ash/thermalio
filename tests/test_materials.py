import numpy as np
import pytest
from heat_solver.materials import process_alpha

def test_process_alpha_scalar_float():
    alpha = 2.5
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    Y = np.array([[0.0, 1.0], [2.0, 3.0]])
    res = process_alpha(alpha, X, Y)
    assert res.shape == (2, 2, 2, 2)
    assert np.allclose(res[0, 0], [[2.5, 0.0], [0.0, 2.5]])

def test_process_alpha_constant_tensor():
    alpha = [[2.0, 0.5], [0.5, 1.0]]
    X = np.array([0.0, 1.0])
    Y = np.array([0.0, 1.0])
    res = process_alpha(alpha, X, Y)
    assert res.shape == (2, 2, 2)
    assert np.allclose(res[0], [[2.0, 0.5], [0.5, 1.0]])

def test_process_alpha_spatially_varying_scalar():
    X = np.array([1.0, 2.0])
    Y = np.array([1.0, 2.0])
    alpha = X * Y
    res = process_alpha(alpha, X, Y)
    assert res.shape == (2, 2, 2)
    assert np.allclose(res[0], [[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(res[1], [[4.0, 0.0], [0.0, 4.0]])

def test_process_alpha_callable_tensor():
    def alpha_func(x, y):
        # returns diagonal tensor proportional to x
        res = np.zeros(x.shape + (2, 2))
        res[..., 0, 0] = x
        res[..., 1, 1] = y
        return res
        
    X = np.array([1.0, 2.0])
    Y = np.array([3.0, 4.0])
    res = process_alpha(alpha_func, X, Y)
    assert res.shape == (2, 2, 2)
    assert np.allclose(res[0], [[1.0, 0.0], [0.0, 3.0]])
    assert np.allclose(res[1], [[2.0, 0.0], [0.0, 4.0]])

def test_process_alpha_invalid_shape():
    alpha = np.ones((3, 3))
    X = np.array([1.0])
    Y = np.array([1.0])
    with pytest.raises(ValueError, match="Invalid alpha shape"):
        process_alpha(alpha, X, Y)
