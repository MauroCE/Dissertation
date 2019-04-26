import numpy as np
from numpy.random import binomial
import matplotlib.pyplot as plt


def sigmoid(x):
    """
    Sigmoid function that can handle arrays. For a thorough discussion on
    speeding up sigmoid calculations, see:
    https://stackoverflow.com/a/25164452/6435921

    :param x: Argument for sigmoid function. Can be float, list or numpy array.
    :type x: float or iterable.
    :return: Evaluation of sigmoid at x.
    :rtype: float
    """
    return 1.0 / (1.0 + np.exp(-x))


def logit(x):
    """
    Logit function. Inverse of sigmoid function. It is optimized using the same
    tricks as in sigmoid().

    :param x: Argument for logit.
    :return: Value of logit at x.
    """
    return np.log(x / (1.0 - x))


def lambda_func(x):
    """
    Lambda function as defined for variational inference.
    :param x: Argument of lambda function. Usually xi_i
    :return: Value of lambda function at xi_i = x.
    """
    return (sigmoid(x) - 0.5) / (2.0*x)


def log_normal_kernel(t, m, v):
    """
    This is basically the argument in the exponential of a normal distribution
    with mean=m and variance=v.
    """
    return -0.5*np.log(2.0 * np.pi*m) - (((t - m) ** 2.0) / (2.0 * v))


def generate_bernoulli(size, params):
    """
    Simulates a Bernoulli data set. Parameters used to generate probabilities.
    params has to be a numpy array in order b0, b1, b2 ...
    """
    # Design matrix X of dimension (n x p)
    nx = len(params) - 1  # n of params - 1  = n of explanatory vars
    design = np.random.normal(loc=0, scale=1, size=(size, nx))
    design = np.hstack((np.ones((size, 1)), design))  # intercept column
    # Bernoulli RVs with inverse logit (sigmoid) of linear predictor as prob
    bin_samples = binomial(n=1, p=sigmoid(np.dot(design, params)), size=size)
    return design, bin_samples


def log_posterior(beta):
    """
    Numerically stable version of the log-posterior. Normalization constants
    have been removed prior to log() transformation, as they would cancel out
    anyways in Metropolis-Hastings.

    :param beta:
    :return:
    """
    # Find the prior distribution. Mean 0, Variance-Covariance 4n(X^T X)^-1
    xbeta = np.dot(X, beta)
    prior = - (0.125 * np.dot(xbeta, xbeta)) / n
    log = np.dot(beta, np.sum(y.reshape(-1, 1) * X, axis=0)) - np.sum(np.log(1 + np.exp(xbeta)))
    return prior + log


def surface_plot(func, xleft, xright, yleft, yright, num):
    """
    Given a R2 -> R function "func" which takes an array of dimension 2 and returns a scalar,
    this function will plot a surface plot over the specified region.
    """
    b1 = np.linspace(xleft, xright, num)
    b2 = np.linspace(yleft, yright, num)
    B1, B2 = np.meshgrid(b1, b2)
    Zbeta = np.array([func(beta) for beta in zip(B1.ravel(), B2.ravel())])
    Zbeta = Zbeta.reshape(B1.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(B1, B2, Zbeta)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return B1, B2, Zbeta


if __name__ == "__main__":
    pass
