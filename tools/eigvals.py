
import numpy as np
import matplotlib.pyplot as plt

def power_method(operator, N, rayleigh_op=None, num_iter=1000):
    """
    Using the power method, iteratively estimates the eigenvector with the largest eigenvalue.

    Arguments:
        operator: Callable. Linear operator from R^N -> R^N, e.g. a function x -> A*x
        N: Int. Dimensionality of eigenvector.
        rayleigh_op: Callable. Optional. If given, calculates the corresponding eigenvalue as well.
        num_iter: Optional. Number of iterations to do before returning

    Returns:
        Tuple of eigenvector and eigenpair. The latter is None unless the rayleigh_op is given.
    """

    x = np.random.random(N)
    x /= np.linalg.norm(x)

    for _ in range(num_iter):
        x = operator(x)
        x /= np.linalg.norm(x)

    eigval = rayleigh_op(x)



    return x, eigval


def matrix_function(A):
    """Creates the function x -> Ax.

    Arguments:
        A: 2d array

    Returns:
        The function x -> Ax."""

    def operator(x):
        return A @ x

    return operator


def create_rayleigh(A):
    """Creates the function x -> (x* A x)/(x* x).
 
    Arguments:
        A: 2d array

    Returns:
        The function calculating the Rayleigh coefficient, given an x"""

    def operator(x):
        return (x.conj().T @ A @ x)/(x @ x)

    return operator
