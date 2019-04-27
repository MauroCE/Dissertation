from utility_functions import sigmoid, logit, log_normal_kernel, lambda_func
from scipy.stats import norm
from scipy.special import beta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap


class NoExplanatoryVariables:
    """
    This class gathers together various stuff relating to the situation in
    which we only have y_1, ... , y_n that are Bernoulli(p) and we have no
    explanatory variables. The true distribution can be found as the likelihood
    and it turns out to be a Beta(a, b).
    We can then write the pdf for theta = logit(p) using transformation of
    random variables. By taking the log we get the true log posterior
    distribution.
    """
    def __init__(self):
        self.n = n
        # Mean of observations
        self.ybar = ybar
        # Sum of observations
        self.sum = self.n*self.ybar
        # Parameters for beta distribution.
        self.a = self.sum + 1
        self.b = self.n - self.sum + 1
        self.true_log_mode = self.a / (self.a + self.b)  # found by derivative
        # Laplace mean and sigma squared
        self.lap_mean = logit(self.ybar)
        self.lap_sigma2 = 1.0 / (self.sum * (1.0 - self.ybar))
        # Variational mean and sigma squared
        self.var_mean, self.var_sigma2 = self._variational_em()

    def true_posterior(self, t):
        """Transformation of beta distribution (not a beta anymore)"""
        return (sigmoid(t)**self.a)*(sigmoid(-t)**self.b)/beta(self.a, self.b)

    def true_log_posterior(self, t):
        """Log of the true posterior"""
        return t*self.a - (self.n + 2)*np.log(1 + np.exp(t))

    def log_likelihood(self, t):
        """log likelihood written in terms of theta. Corresponds to log
        posterior"""
        return t * np.sum(y) - self.n * np.log(1 + np.exp(t))

    def likelihood(self, t):
        """likelihood written in terms of theta. Corresponds to posterior. """
        # TODO: Not working somehow
        return np.exp(t * np.sum(y)) / ((1 + np.exp(t)) ** self.n)

    def laplace(self, t):
        """Laplace approximation done on the likelihood only."""
        return norm(loc=self.lap_mean, scale=np.sqrt(self.lap_sigma2)).pdf(t)

    def log_laplace(self, t):
        """Log of the laplace approximation."""
        return log_normal_kernel(t, self.lap_mean, self.lap_sigma2)

    def _variational_em(self):
        """Internal function to find mean and variance for variational em."""
        xi_list = np.random.uniform(size=self.n)  # try 0, 1? Low1, High2
        # number of EM iterations
        for i in range(10):
            for xi_index in range(len(xi_list)):
                xi_list[xi_index] = np.sqrt(
                    (1.0 + self.sum - 0.5*self.n)/(2.0*np.sum(lambda_func(xi_list)))
                )
        # once parameters have been optimized, use them to find mean and var
        sigma2 = 1.0 / (2.0*np.sum(lambda_func(xi_list)))
        mean = sigma2*self.n*(self.ybar - 0.5)
        return mean, sigma2

    def variational(self, t):
        """Variational normal distribution"""
        return norm(loc=self.var_mean, scale=np.sqrt(self.var_sigma2)).pdf(t)

    def log_variational(self, t):
        """Log Variational distribution"""
        return log_normal_kernel(t, self.var_mean, self.var_sigma2)

    def compare_approximations(self, n_min, n_max, n_step):
        """This function tries to compare Laplace approximation and
        variational approximation. Since they are both normal, we only really
        need to compare their mean and their scale. For this reason, this
        function tries to create some useful plots."""
        var_means = []
        var_scales = []
        lap_means = []
        lap_scales = []
        n_list = list(range(n_min, n_max, n_step))
        for n in n_list:
            self.n = n
            self.ybar = np.mean(
                np.random.binomial(n=1, p=sigmoid(theta), size=self.n)
            )
            self.sum = self.n * self.ybar
            lap_means.append(logit(self.ybar))
            lap_scales.append(np.sqrt(1.0 / (self.sum * (1.0 - self.ybar))))
            var_mean, var_sigma2 = self._variational_em()
            var_means.append(var_mean)
            var_scales.append(np.sqrt(var_sigma2))
        var_means = np.array(var_means)
        var_scales = np.array(var_scales)
        lap_means = np.array(lap_means)
        lap_scales = np.array(lap_scales)
        # Make colormaps prettier
        blue_cmap = ListedColormap(cm.Blues(np.linspace(0, 1, 20))[10:, :-1])
        oran_cmap = ListedColormap(cm.Oranges(np.linspace(0, 1, 20))[10:, :-1])

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 10))
        fig.suptitle("Comparing Parameters of Laplace and Variational"
                     " Normal Approximations", fontsize=20)
        # Variational, Laplace sigma
        ax[0].scatter(var_means, var_scales,
                   c=n_list, cmap=oran_cmap)
        ax[0].plot(var_means, var_scales, c='orange', label='variational')
        ax[0].scatter(lap_means, lap_scales,
                   c=n_list, cmap=blue_cmap)
        ax[0].plot(lap_means, lap_scales, c='blue', label='laplace')
        ax[0].set_xlabel(r'$\mu$')
        ax[0].set_ylabel(r'$\sigma$')
        ax[0].axvline(theta, alpha=0.5,
                      label='True ' + r'$\theta$', ls=':')
        ax[0].legend()
        # Difference in mean, difference in scales
        ax[1].scatter(var_means - lap_means, var_scales - lap_scales,
                   c=n_list, cmap=oran_cmap)
        ax[1].plot(var_means - lap_means, var_scales - lap_scales,
                   c='orange', label='Variational - Laplace')
        ax[1].legend()
        ax[1].set_xlabel(r'$\mu_V - \mu_L$')
        ax[1].set_ylabel(r'$\sigma_V - \sigma_L$')
        # First plot annotations
        for i, txt in enumerate(n_list):
            ax[0].annotate(txt, (var_means[i], var_scales[i]))
            ax[0].annotate(txt, (lap_means[i], lap_scales[i]))
            ax[1].annotate(txt, (var_means[i] - lap_means[i],
                                 var_scales[i] - lap_scales[i]))
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig("images/comparing_no_exp_approximations.png")
        plt.show()


if __name__ == "__main__":
    # pretty grid, random seed for consistency (?), # data points, create data
    sns.set_style('darkgrid')
    np.random.seed(1)
    n = 100
    theta = 1
    y = np.random.binomial(n=1, p=sigmoid(theta), size=n)
    ybar = np.mean(y)
    # Different plotting scales for posterior and log posterior
    x = np.linspace(0, 2, 100)
    x_log = np.linspace(-10, 10, 100)
    # Instantiate model with no explanatory variables
    model = NoExplanatoryVariables()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    # Normal scale
    ax[0].plot(x, model.true_posterior(x), label='true posterior')
    ax[0].plot(x, model.laplace(x), label='laplace')
    ax[0].plot(x, model.variational(x), label='variational')
    ax[0].legend()
    ax[0].set_title('Posterior Scale')
    # Log scale
    ax[1].plot(x_log, model.true_log_posterior(x_log), label='true log post')
    ax[1].plot(x_log, model.log_likelihood(x_log), label='log likelihood')
    ax[1].plot(x_log, model.log_laplace(x_log), label='log laplace')
    ax[1].plot(x_log, model.log_variational(x_log), label='log variational')
    ax[1].set_title('Log Posterior Scale')
    ax[1].axvline(model.true_log_mode, alpha=0.5, ls=':', label='log mode')
    ax[1].legend()
    plt.savefig("images/no_explanatory.png")
    plt.show()
    # Compare mean and scale of Laplace and Variational normal approximations
    model.compare_approximations(1000, 10000, 1000)
