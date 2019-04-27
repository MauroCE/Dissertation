import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, multivariate_normal, norm
from utility_functions import generate_bernoulli, \
    log_normal_kernel, metropolis_multivariate, sigmoid, lambda_func, \
    row_outer, prepare_surface_plot, generate_subplots


sns.set_style('darkgrid')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 6)
plt.rc('figure', titlesize=14)


class ExplanatoryVariables:
    """
    Contains functions for the scenario where we have explanatory variables.
    """
    def __init__(self, n, params, X, y, mu0, sigma0):
        # Might need to feed n and params into __init__().
        self.n = n
        self.params = params
        self.p = len(self.params)  # Number of parameters
        self.X = X
        self.y = y
        # mean and variance-covariance matrix for Normal PRIOR
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.sigma0_inv = np.linalg.inv(self.sigma0)
        # Find mode of the log posterior and the observed information matrix
        self.mode, self.hess_inv = self._find_mode_and_hess_inv()
        # Find Variance_Covariance matrix for Laplace
        self.lap_vcov = self._laplace_vcov()
        # Variational mean and vcov
        self.var_mean, self.var_vcov = self._variational_em()
        # mcmc
        self.kdes = []
        self.samples = []
        self.ar = None  # Acceptance rate

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
        samples, ar = metropolis_multivariate(
            p=self.log_posterior,
            z0=self.mode,
            cov=self.hess_inv,
            n_samples=s,
            burn_in=b,
            thinning=t
        )
        self.samples = samples
        self.ar = ar
        return self.samples

    def mcmc_timeseries(self):
        """Static method to plot time series of mcmc samples. It works
        independently of the number of parameters."""
        fig, ax = plt.subplots(nrows=self.p, ncols=1,
                               figsize=(13, 3*self.p))
        for i in range(self.p):
            ax[i].plot(self.samples[:, i], 'k')
            ax[i].set_xlabel("Iteration", fontsize=12)
            ax[i].set_ylabel(r'$\beta_{{{}}}$'.format(i + 1), fontsize=12)
        fig.suptitle(r"MCMC Time series plots", fontsize=20)
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        plt.show()

    def mcmc_kde_on_hist(self):
        """Plots a histogram with KDE for each parameter, from a
        mcmc sample."""
        # Instantiate figure
        fig, ax = plt.subplots(self.p, figsize=(10, 4 * self.p))
        # store all kdes to return them and use them later
        kdes = []
        # Loop to do all the plots
        for p in range(self.p):
            kde = gaussian_kde(self.samples[:, p].reshape(1, -1))
            kdes.append(kde)
            x_values = np.linspace(min(self.samples[:, p]),
                                   max(self.samples[:, p]), 100)
            ax[p].hist(self.samples[:, p], bins=500, density=True,
                       label=r'$p(\beta_{{{}}} \mid x)$'.format(p + 1))
            ax[p].plot(x_values, kde.evaluate(x_values), label='kde')
            ax[p].legend()
        plt.show()
        # save them
        self.kdes = kdes
        return kdes

    def _laplace_vcov(self):
        """Finds the variance-covariance matrix of Laplace approximation"""
        # Find laplace variance-covariance matrix
        pi = sigmoid(np.dot(self.X, self.mode))
        outed = row_outer(self.X)
        return np.linalg.inv(
            self.sigma0_inv +
            (((pi*(1 - pi)).reshape(self.n, 1, 1)) * outed).sum(axis=0)
        )

    def log_laplace(self, beta):
        """Evaluates laplace approximation at beta."""
        return multivariate_normal(mean=self.mode,
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
        fig.suptitle("Contour and Surface plot of Laplace and Variational")
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
        # Contour plots
        ax = fig.add_subplot(122)
        lap_contour = ax.contour(xl, yl, zl, colors='orange', linestyles='--')
        pos_contour = ax.contour(x, y, z, colors='blue', linestyles='-.')
        var_contour = ax.contour(xv, yv, zv, colors='green', linestyles=':')
        lap_handle, _ = lap_contour.legend_elements()
        pos_handle, _ = pos_contour.legend_elements()
        var_handle, _ = var_contour.legend_elements()
        lap_index = len(lap_handle) // 2
        pos_index = len(pos_handle) // 2
        var_index = len(var_handle) // 2
        ax.legend([lap_handle[lap_index], pos_handle[pos_index],
                   var_handle[var_index]],
                  ['Log Laplace', 'Log Posterior', 'Log-Variational'],
                  loc='center')
        plt.show()

    def marginal_plots(self):
        """A plot for each parameter containing marginal laplace, marginal
          variational and kde on marginal samples."""
        # Laplace Marginals
        laplace_marginals = [
            norm(loc=self.mode[p], scale=np.sqrt(self.lap_vcov[p, p]))
            for p in range(self.p)
        ]
        # Variational Marginals
        var_marginals = [
            norm(loc=self.var_mean[p], scale=np.sqrt(self.var_vcov[p, p]))
            for p in range(self.p)
        ]
        # Plot
        fig, axes = generate_subplots(
            self.p, row_wise=True,
            suptitle="Dataset size: {}, Number of "
                     "Parameters: {}".format(self.n, self.p), fontsize=20)
        for p, ax in zip(np.arange(self.p), axes):
            # obtain all marginals / kdes needed
            laplace, variational, mcmc = laplace_marginals[p], var_marginals[
                p], self.kdes[p]
            # Use standard deviation to choose an x plotting range that
            # makes the plot pretty
            max_std = max(laplace.std(), variational.std(),
                          np.sqrt(mcmc.covariance[0, 0]))
            xmin = self.mode[p] - 4 * max_std
            xmax = self.mode[p] + 4 * max_std
            x_values = np.linspace(xmin, xmax, 200)
            # plot laplace, variational, mcmc
            ax.plot(x_values, laplace.pdf(x_values), label='laplace_marginal')
            ax.plot(x_values, mcmc.pdf(x_values), label='mcmc kde')
            ax.plot(x_values, variational.pdf(x_values),
                    label='variational marginal')
            ax.set_xlabel(r'$\beta_{{{}}}$'.format(p + 1))
            ax.legend()
            ax.set_title(r"KDE and Marginals for $\beta_{{{}}}$".format(p + 1))
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
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
    mcmc_samples = model.sample(2000, 500, 10)
    print("MH acceptance rate: {:.3}".format(model.ar))
    model.mcmc_timeseries()
    model.mcmc_kde_on_hist()
    # to plot posterior nicely find correct bounds from mcmc samples
    b1min = min(mcmc_samples[:, 0])
    b1max = max(mcmc_samples[:, 0])
    b2min = min(mcmc_samples[:, 1])
    b2max = max(mcmc_samples[:, 1])
    # Surface plot
    model.surface_plots(b1min, b1max, b2min, b2max, 200)
    # Marginal Plot
    model.marginal_plots()
