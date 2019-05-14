from utility_functions import sigmoid, logit, log_normal_kernel, lambda_func,\
    blue_cmap, oran_cmap, setup_plotting, metropolis, mkdir_p
from scipy.stats import norm, gaussian_kde
from scipy.special import beta
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns


setup_plotting()

#sns.set_style('white')


class NoExplanatoryVariables:
    """
    This class gathers together various stuff relating to the situation in
    which we only have y_1, ... , y_n that are Bernoulli(p) and we have no
    explanatory variables. The true distribution can be found as the likelihood
    and it turns out to be a Beta(a, b).
    We can then write the pdf for beta = logit(p) using transformation of
    random variables. By taking the log we get the true log posterior
    distribution.
    """
    def __init__(self, save=True, dict=None):
        # save seed number for saving
        self.seed = dict['seed']
        self.n = dict['n']
        self.beta = dict['beta']  # beta (the true one)
        # Mean of observations
        self.ybar = dict['ybar']
        # Sum of observations
        self.sum = self.n*self.ybar
        # Parameters for beta distribution.
        self.a = self.sum
        self.b = self.n - self.sum
        # Laplace mean and sigma squared
        self.lap_mean = logit(self.ybar)
        self.lap_sigma2 = 1.0 / (self.sum * (1.0 - self.ybar))
        # Variational mean and sigma squared
        self.var_mean, self.var_sigma2 = self._variational_em()
        # mcmc samples and stuff
        self.samples = np.array([])
        self.ar = None  # Acceptance rate
        # Whether to save images or not
        self.save = save
        # scale factor
        self.mcmc_scale_factor = None
        # kde
        self.kde = None
        self.kde_scale = None
        # true mode and true hessian inverse
        self.sample_mode, self.sample_hess_inv = self._find_true_mode()
        #print("n: ", self.n)
        #print("seed: ", self.seed)
        #print("-"*40)
        #print("True beta: ", self.beta)
        #print("True E[Y]: ", sigmoid(self.beta))
        #print("True mode: ", logit(sigmoid(self.beta)))
        #print("Sample beta: ", logit(self.ybar))
        #print("Sample mean: ", self.ybar)
        #print("Sample mode: ", logit(self.ybar))

    def _find_true_mode(self):
        """Finds the true mode of the log-posterior"""
        result = minimize(lambda x: -self.true_log_posterior(x),
                          np.array([0]))
        return result.x, result.hess_inv

    def save_image(self, image_name):
        """Creates directory where we can save images."""
        # create dir name. If kde_scale is given save based on that
        dir_name = "n_{}_sd_{}".format(
            self.n,
            self.seed
        )
        # Try to save, but might need to create directory first
        try:
            plt.savefig("images/no_explanatory/{}/{}".format(
                dir_name,
                image_name)
            )
        except FileNotFoundError:
            mkdir_p("images/no_explanatory/{}".format(dir_name))
            plt.savefig("images/no_explanatory/{}/{}".format(
                dir_name,
                image_name)
            )
        return

    def true_posterior(self, t):
        """Transformation of beta distribution (not a beta anymore)"""
        return (sigmoid(t)**self.a)*(sigmoid(-t)**self.b)/beta(self.a, self.b)

    def true_log_posterior(self, t):
        """Log of the true posterior"""
        return -np.log(beta(self.a, self.b)) \
            + t*self.a - self.n*np.log(1 + np.exp(t))

    def log_likelihood(self, t):
        """log likelihood written in terms of beta. Corresponds to log
        posterior"""
        return t * np.sum(y) - self.n * np.log(1 + np.exp(t))

    def likelihood(self, t):
        """likelihood written in terms of beta. Corresponds to posterior. """
        # TODO: Not working somehow
        return np.exp(t * np.sum(y)) / ((1 + np.exp(t)) ** self.n)

    def laplace(self, t):
        """Laplace approximation done on the likelihood only."""
        return norm(loc=self.lap_mean, scale=np.sqrt(self.lap_sigma2)).pdf(t)

    def log_laplace(self, t):
        """Log of the laplace approximation."""
        return norm(loc=self.lap_mean, scale=np.sqrt(self.lap_sigma2)).logpdf(t)

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
        return norm(loc=self.var_mean, scale=np.sqrt(self.var_sigma2)).logpdf(t)

    def sample(self, s, b, t, scale=1.0, kde_scale=None):
        """RWM algorithm"""
        # save scale for plotting
        self.mcmc_scale_factor = scale
        samples, ar = metropolis(
            p=self.true_log_posterior,
            z0=np.array([self.lap_mean]),
            cov=1,
            n_samples=s,
            burn_in=b,
            thinning=t,
            scale=scale
        )
        self.samples = samples
        self.ar = ar
        # find also KDE
        self.kde = gaussian_kde(self.samples.flatten())
        if kde_scale is not None:
            self.kde_scale = kde_scale
            self.kde.set_bandwidth(bw_method=kde_scale)
        print("MH acceptance rate: {:.3}".format(self.ar))
        return self.samples

    def mcmc_kde(self, t):
        """Can be called evaluate kde found on histogram of mcmc samples."""
        return self.kde.pdf(t)

    def plot_densities(self, xmin, xmax, xstep, xlogmin, xlogmax, xlogstep):
        """Plot1: posterior, laplace, variational.
           Plot2: log-post, log-laplace, log-variational."""
        x = np.linspace(xmin, xmax, xstep)
        x_log = np.linspace(xlogmin, xlogmax, xlogstep)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        # Normal scale
        ax[0].plot(x, self.true_posterior(x), label='true posterior', color='b', lw=1.0)
        #ax[0].fill_between(x, self.true_posterior(x), color='k', alpha=0.2, lw=1.5)
        ax[0].plot(x, self.laplace(x), label='laplace',ls='--', dashes=(5, 5),
                   color='#ff7f0e', lw=2.0)
        ax[0].plot(x, self.variational(x), label='variational', ls='--', dashes=(5, 5),
                   color='#2ca02c', lw=2.0)
        ax[0].set_xlabel(r'$\beta$')
        ax[0].set_title('Posterior Scale')
        #ax[0].axvline(self.sample_mode, color='k', ls=':', label="Sample Mode", alpha=0.2)
        #ax[0].axvline(self.beta, color='k', ls='-.', label="Pop Mode", alpha=0.2)
        ax[0].legend()
        # Log scale
        ax[1].plot(x_log, self.true_log_posterior(x_log),
                   label='true log post', color='b', lw=1.0)
        ax[1].plot(x_log, self.log_laplace(x_log), label='log laplace',
                   ls='--', dashes=(5, 5), color='#ff7f0e', lw=2.0)
        ax[1].plot(x_log, self.log_variational(x_log),
                   label='log variational', ls='--', dashes=(5, 5),
                   color='#2ca02c', lw=2.0)
        ax[1].set_title('Log Posterior Scale')
        ax[1].set_xlabel(r'$\beta$')
        #ax[1].axvline(self.sample_mode, color='k', ls=':', label="Sample Mode", alpha=0.2)
        #ax[1].axvline(self.beta, color='k', ls='-.', label='Pop Mode', alpha=0.2)
        ax[1].legend()
        if self.save:
            self.save_image("posterior_laplace_variational.png")
        plt.show()

    def plot_vertical_distance(self, xmin, xmax, xstep, xlogmin, xlogmax, xlogstep):
        """Plots vertical distance between true posterior and the
        approximations."""
        # X data
        x = np.linspace(xmin, xmax, xstep)
        x_log = np.linspace(xlogmin, xlogmax, xlogstep)
        # Figure
        fig, ax = plt.subplots(nrows=1, ncols=2)
        # normal scale, no abs
        # normal scale, abs
        tv_abs = abs(self.true_posterior(x) - self.variational(x))
        tl_abs = abs(self.true_posterior(x) - self.laplace(x))
        ax[0].plot(x, tv_abs, label=r'$\left| p(\beta\mid \mathbf{y})- q_{v}(\beta)\right|$')
        ax[0].plot(x, tl_abs, label=r'$\left| p(\beta\mid \mathbf{y})- q_{l}(\beta)\right|$')
        ax[0].axvline(self.sample_mode, color='k', ls='--', alpha=0.4,
                      label="True Mode")
        ax[0].axvline(self.ybar, color='g', ls=':', alpha=0.4, label=r'$\overline{y}$')
        ax[0].legend()
        ax[0].set_xlabel(r'$\beta$')
        # log scale, abs
        ax[1].plot(x_log, abs(self.true_log_posterior(x_log) - self.log_variational(x_log)), label="|logtrue-logvar|")
        ax[1].plot(x_log, abs(self.true_log_posterior(x_log) - self.log_laplace(x_log)), label="|logtrue-loglap")
        #ax[1].axvline(self.sample_mode, color='k', ls='--', alpha=0.4, label="True Mode")
        #ax[1].axvline(self.ybar, color='g', ls=':', alpha=0.4, label=r'$\overline{y}$')
        ax[1].legend()
        ax[1].set_xlabel(r'$\beta$')
        return fig, ax, x, x_log, tv_abs, tl_abs

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
        number = len(n_list)
        # store previous n to save figure properly
        n_old = self.n
        for n in n_list:
            self.n = n
            self.ybar = np.mean(
                np.random.binomial(n=1, p=sigmoid(b), size=self.n)
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
        # scales
        s_sizes = np.arange(10, number*11, 10)
        fig, ax = plt.subplots()
        ax.set_title("Comparing Parameters of Laplace and Variational"
                     " Normal Approximations")
        # Variational, Laplace sigma
        ax.scatter(var_means, var_scales, c='#2ca02c', s=s_sizes)  # c=n_list
        ax.plot(var_means, var_scales, c='#2ca02c', label='variational')
        ax.scatter(lap_means, lap_scales, c='#ff7f0e', s=s_sizes)
        ax.plot(lap_means, lap_scales, c='#ff7f0e', label='laplace')
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\sigma$')
        ax.axvline(self.beta, alpha=0.5,
                   label='True Mode', ls=':', color='k')
        ax.legend()
        # First plot annotations
        for i, txt in enumerate(n_list):
            #ax.annotate(txt, (var_means[i], var_scales[i]))
            ax.annotate(txt, (lap_means[i], lap_scales[i]))
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        if self.save:
            self.n = n_old
            self.save_image("mean_sd_comparison.png")
        plt.show()


if __name__ == "__main__":
    n = 300  # 100
    b = 1  # beta
    seed = 24 #np.random.randint(0, 100)  # 40  # 88
    print('seed: ', seed)
    np.random.seed(seed)
    y = np.random.binomial(n=1, p=sigmoid(b), size=n)
    ybar = np.mean(y)
    dict = {
        'n': n,
        'ybar': ybar,
        'seed': seed,
        'beta': b
    }
    # Instantiate model with no explanatory variables
    model = NoExplanatoryVariables(save=True, dict=dict)
    # Sample the model and get kde
    # samples = model.sample(s=100000, b=500, t=1, scale=0.25, kde_scale=0.15)

    # find plotting range
    shift = np.sqrt(model.sample_hess_inv[0][0]) * 3
    xmin = model.sample_mode - shift
    xmax = model.sample_mode + shift
    print(xmin, xmax)
    # Plot densities
    model.plot_densities(xmin, xmax, 500, xmin - 14 * shift, xmax + 14 * shift, 500) # 0,2,100,-10,10,100
    print("Var mean: ", model.var_mean)
    print("Lap mean: ", model.lap_mean)
    # vertical distances
    #model.plot_vertical_distance(0, 2, 100, 0, 2, 100)
    # Compare mean and scale of Laplace and Variational normal approximations
    #model.compare_approximations(1000, 18000, 2000) #1000, 10000, 1000
    model.compare_approximations(100, 1000, 100)
