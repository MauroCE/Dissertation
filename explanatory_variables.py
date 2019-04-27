import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, multivariate_normal
from utility_functions import generate_bernoulli, surface_plot, \
    log_normal_kernel, metropolis_multivariate, sigmoid, lambda_func, \
    row_outer, prepare_surface_plot


sns.set_style('darkgrid')


class ExplanatoryVariables:
    """
    Contains functions for the scenario where we have explanatory variables.
    """
    def __init__(self, n, params, X, y, mu0, sigma0):
        # Might need to feed n and params into __init__().
        self.n = n
        self.params = params
        self.n_params = len(self.params)
        self.X = X
        self.y = y
        # mean and variance-covariance matrix for Normal PRIOR
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.sigma0_inv = np.linalg.inv(self.sigma0)
        # Find mode of the log posterior and the observed information matrix
        self.log_post_mode, self.hess_inv = self._find_mode_and_hess_inv()
        # Find Variance_Covariance matrix for Laplace
        self.lap_vcov = self._laplace_vcov()
        # Variational mean and vcov
        self.var_mean, self.var_vcov = self._variational_em()

    def _find_mode_and_hess_inv(self):
        """Finds mode and inverse hessian of log posterior."""
        # Minimize the function
        result = minimize(
            fun=lambda x: -self.log_posterior(x),
            x0=np.zeros(len(self.params))
        )
        # return mode and inverse hessian
        return result.x, result.hess_inv

    def log_posterior(self, beta):
        """Log posterior."""
        return np.sum(self.y*np.dot(self.X, beta)) \
               - np.log(1 + np.exp(np.dot(self.X, beta))).sum() \
               + log_normal_kernel(beta, self.mu0, self.sigma0,
                                   multivariate=True)

    def sample(self, s, b=0, t=1):
        """
        Samples the log-posterior distribution using Random-Walk Metropolis
        Hastings.

        :param s: Number of total final samples.
        :param b: Number of burn-in samples.
        :param t: Thinning.
        :return: samples[burn_in::thinning], acceptance_rate
        """
        return metropolis_multivariate(
            p=self.log_posterior,
            z0=self.log_post_mode,
            cov=self.hess_inv,
            n_samples=s,
            burn_in=b,
            thinning=t
        )

    @staticmethod
    def mcmc_timeseries(samples):
        """Static method to plot time series of mcmc samples. It works
        independently of the number of parameters."""
        n_params = samples.shape[1]
        fig, ax = plt.subplots(nrows=n_params, ncols=1,
                               figsize=(13, 3*n_params))
        for i in range(n_params):
            ax[i].plot(samples[:, i], 'k')
            ax[i].set_xlabel("Iteration", fontsize=12)
            ax[i].set_ylabel(r'$\beta_{{{}}}$'.format(i + 1), fontsize=12)
        fig.suptitle(r"MCMC Time series plots", fontsize=20)
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        plt.show()

    @staticmethod
    def mcmc_kde_on_hist(samples):
        """Plots a histogram with KDE for each parameter, from a
        mcmc sample."""
        n_params = samples.shape[1]
        # Instantiate figure
        fig, ax = plt.subplots(n_params, figsize=(10, 4 * n_params))
        # store all kdes to return them and use them later
        kdes = []
        # Loop to do all the plots
        for p in range(n_params):
            kde = gaussian_kde(samples[:, p].reshape(1, -1))
            kdes.append(kde)
            x_values = np.linspace(min(samples[:, p]), max(samples[:, p]), 100)
            ax[p].hist(samples[:, p], bins=500, density=True,
                       label=r'$p(\beta_{{{}}} \mid x)$'.format(p + 1))
            ax[p].plot(x_values, kde.evaluate(x_values), label='kde')
            ax[p].legend()
        plt.show()
        return kdes

    def _laplace_vcov(self):
        """Finds the variance-covariance matrix of Laplace approximation"""
        # Find laplace variance-covariance matrix
        pi = sigmoid(np.dot(self.X, self.log_post_mode))
        outed = row_outer(self.X)
        return np.linalg.inv(
            self.sigma0_inv +
            (((pi*(1 - pi)).reshape(self.n, 1, 1)) * outed).sum(axis=0)
        )

    def log_laplace(self, beta):
        """Evaluates laplace approximation at beta."""
        return multivariate_normal(mean=self.log_post_mode,
                                   cov=self.lap_vcov).logpdf(beta)

    def _var_params(self, xi_vector):
        """Finds the variance-covariance matrix and mean for the variational
        normal approximation."""
        # find variance-covariance matrix
        lamb = lambda_func(xi_vector).reshape(self.n, 1, 1)
        outed = row_outer(self.X)
        vcov = np.linalg.inv(
            (2*lamb*outed).sum(axis=0) +
            self.sigma0_inv
        )
        # find mean vector
        first = np.dot(self.sigma0_inv, self.mu0)
        mean = np.dot(
            vcov,
            ((self.y - 0.5).reshape(-1, 1)*self.X).sum(axis=0) + first
        )
        return mean, vcov

    def _variational_em(self):
        """
        This version optimizes each parameter in turn.
        """
        xi = np.random.rand(self.n)
        for i in range(10):
            for index, xin in enumerate(xi):
                xn = self.X[index]
                # mean and vcov of posterior over params
                mn, sn = self._var_params(xi)
                # now take only positive values of variational parameter
                xin = np.sqrt(np.dot(xn, np.dot((sn + np.outer(mn, mn)), xn)))
                xi[index] = xin
        return mn, sn

    def log_variational(self, beta):
        """Evaluates variational approximation at beta."""
        return multivariate_normal(mean=self.var_mean,
                                   cov=self.var_vcov).logpdf(beta)

    def surface_plots(self, xmin, xmax, ymin, ymax, n):
        """Plots surfaces of log posterior, laplace and variational."""
        args = [xmin, xmax, ymin, ymax, n]
        # Get grid data for log posterior, laplace and variational
        x, y, z = prepare_surface_plot(self.log_posterior, *args)
        xv, yv, zv = prepare_surface_plot(self.log_variational, *args)
        xl, yl, zl = prepare_surface_plot(self.log_laplace, *args)
        # need to rescale data to make it ""normalized""
        zv = zv - np.max(zv)
        zl = zl - np.max(zl)
        z = z - np.max(z)
        # Put plot together
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        # Legend needs some manual modifications
        p = ax.plot_surface(x, y, z, label='log-posterior')
        l = ax.plot_surface(xl, yl, zl, label='log-laplace')
        v = ax.plot_surface(xv, yv, zv, label='log-variational')
        p._facecolors2d, p._edgecolors2d = p._facecolors3d, p._edgecolors3d
        l._facecolors2d, l._edgecolors2d = l._facecolors3d, l._edgecolors3d
        v._facecolors2d, v._edgecolors2d = v._facecolors3d, v._edgecolors3d
        # Finish off with labels
        ax.legend()
        ax.set_xlabel(r'$\beta_1$')
        ax.set_ylabel(r'$\beta_2$')
        ax.set_title("Variational vs Laplace", fontsize=13)
        # Contour plots
        ax = fig.add_subplot(122)
        lap_contour = ax.contour(xl, yl, zl, cmap=cm.Blues)
        pos_contour = ax.contour(x, y, z, cmap=cm.Oranges)
        var_contour = ax.contour(xv, yv, zv, cmap=cm.Greens)
        lap_handle, _ = lap_contour.legend_elements()
        pos_handle, _ = pos_contour.legend_elements()
        var_handle, _ = var_contour.legend_elements()
        lap_index = len(lap_handle) // 2
        pos_index = len(pos_handle) // 2
        var_index = len(var_handle) // 2
        ax.legend([lap_handle[lap_index], pos_handle[pos_index],
                   var_handle[var_index]],
                  ['Log Laplace', 'Log Posterior', 'Log-Variational'])
        plt.show()


if __name__ == "__main__":
    # Settings
    n = 1000
    params = [1.0, 0.5]
    X, y = generate_bernoulli(n, params)
    # Initialize model
    model = ExplanatoryVariables(
        n=n,
        params=params,
        X=X,
        y=y,
        mu0=np.zeros(len(params)),
        sigma0=4*n*np.linalg.inv(np.dot(X.T, X))
    )
    # time series of mcmc
    mcmc_samples, acceptance_rate = model.sample(200, 500, 10)
    model.mcmc_timeseries(mcmc_samples)
    model.mcmc_kde_on_hist(mcmc_samples)
    # to plot posterior nicely find correct bounds from mcmc samples
    b1min = min(mcmc_samples[:, 0])
    b1max = max(mcmc_samples[:, 0])
    b2min = min(mcmc_samples[:, 1])
    b2max = max(mcmc_samples[:, 1])
    # Surface plot
    model.surface_plots(b1min, b1max, b2min, b2max, 500)


