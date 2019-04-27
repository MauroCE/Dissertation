import numpy as np
from numpy.random import binomial
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sigmoid(x):
    """
    Sigmoid function that can handle arrays. For a thorough discussion on
    speeding up sigmoid calculations, see:
    https://stackoverflow.com/a/25164452/6435921

    :param x: Argument for sigmoid function. Can be float, list or numpy array.
    :type x: float or iterable.
    :return: Evaluation of sigmoid at x.
    :rtype: np.array
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


def log_normal_kernel(t, m, v, multivariate=False):
    """
    This is basically the argument in the exponential of a normal distribution
    with mean=m and variance=v.
    """
    if multivariate:
        return -0.5*np.dot(t - m, np.dot(np.linalg.inv(v), t - m))
    else:
        return - (0.5*(t - m) ** 2.0) / v


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


def surface_plot(f, xmin, xmax, ymin, ymax, n):
    """
    Given a R2 -> R function "func" which takes an array of dimension 2 and returns a scalar,
    this function will plot a surface plot over the specified region.
    """
    # Get grid data
    x, y, z = prepare_surface_plot(f, xmin, xmax, ymin, ymax, n)
    # Plot 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)
    ax.update({'xlabel': 'x', 'ylabel': 'y', 'zlabel': 'z'})
    return fig, ax, x, y, z


def prepare_surface_plot(f, xmin, xmax, ymin, ymax, n):
    """This function creates the three X, Y and Z arrays that are needed to
    plot surfaces, but it does not plot them. It can be used to put multiple
    plots into one figure."""
    x, y = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))
    z = np.array([f(xy) for xy in zip(x.ravel(), y.ravel())]).reshape(x.shape)
    return x, y, z


def row_outer(A):
    """
    Computes the row-wise outer product a_i a_i^T where each row is a_i.
    """
    return np.einsum('ni,nj->nij', A, A)


def metropolis_multivariate(p, z0, cov, n_samples=100, burn_in=0, thinning=1):
    """
    Random-Walk Metropolis algorithm for a multivariate probability density.

    :param p: Probability distribution we want to sample from.
    :type p: callable
    :param z0: Initial guess for Metropolis algorithm.
    :type z0: np.array
    :param cov: Variance-Covariance matrix for normal proposal distribution.
    :type cov: np.array
    :param n_samples: Number of total samples we want to obtain. This is the
                      number of samples created after applying burn-in and
                      thinning. Basically `samples[burn_in::thinning]`.
    :type n_samples: int
    :param burn_in: Number of samples to burn initially.
    :type: burn_in: int
    :param thinning: If `thinning` > 1 then after applying burn_in we get
                     every <<thinning>> samples.
    :type thinning: int
    :return: `n_samples` samples from `p`.
    :rtype: np.array
    """
    # Initialize algorithm. Calculate num iterations. Acceptance counter.
    z, n_params, pz = z0, len(z0), p(z0)
    tot = burn_in + (n_samples - 1) * thinning + 1
    accepted = 0
    # Init list storing all samples. Generate random numbers.
    sample_list = np.zeros((tot, n_params))
    logu = np.log(np.random.uniform(size=tot))
    normal_shift = multivariate_normal.rvs(mean=np.zeros(n_params),
                                           cov=cov, size=tot)
    for i in range(tot):
        # Sample a candidate from Normal(mu, sigma)
        cand = z + normal_shift[i]

        try:
            # Store values to save computations. logu[i] <= 0 for u \in (0, 1)
            p_cand = p(cand)
            if p_cand - pz > logu[i]:
                z, pz, accepted = cand, p_cand, accepted + 1
        except (OverflowError, ValueError, RuntimeWarning):
            continue

        sample_list[i, :] = z
    return sample_list[burn_in::thinning], accepted / tot



def metropolis(p, scale, z0=None, n_samples=100, burn_in=0, thinning=1, log=False):
    """
    Metropolis algorithm used to sample from a multivariate probability distribution.
    """
    z = z0 if z0 is not None else np.random.uniform()
    pz = p(z)
    # calculate total number of calculations
    tot = burn_in + (n_samples - 1) * thinning + 1
    # store the number of accepted samples to see acceptance rate
    accepted = 0
    # stores all the samples
    sample_list = np.zeros(tot)
    # Generate uniform random numbers outside the loop
    u = np.random.uniform(size=tot)
    # if covariance matrix is not provided, use default ones?
    normal_shift = norm.rvs(loc=0, scale=scale, size=tot)

    for i in range(tot):
        # Sample a candidate from Normal(mu, sigma)
        cand = z + normal_shift[i]
        # Acceptance probability
        try:
            p_cand = p(cand)
            if log:
                prob = min(0, p_cand - pz)
            else:
                prob = min(1, p_cand / pz)  # Notice this is different from min(1, pcand / pz) as we are in logarithmic scale
        except (OverflowError, ValueError, RuntimeWarning):
            continue
        if log:
            condition = prob > np.log(u[i])
        else:
            condition = prob > u[i]
        if condition:  # Notice that this is log(u) because we are in logarithmic scale
            z = cand
            pz = p_cand  # to save computations
            accepted += 1

        sample_list[i] = z

    # Finally want to take every Mth sample in order to achieve independence
    print("Acceptance rate: ", accepted / tot)
    return sample_list[burn_in::thinning]


if __name__ == "__main__":
    pass
