# Import external libraries
import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

# Set parameters to make plots look more professional
sns.set_style('darkgrid')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 6)
plt.rc('figure', titlesize=14)
orange = '#ff7f0e'
blue = '#2ca02c'
grey = "#7f7f7f"
# Set random seed for nicer plots
np.random.seed(4)

# Simulate data
x = np.linspace(0, 10, 1000)
pdf, cdf = np.exp(-x), 1 - np.exp(-x)  # exponential pdf and cdf
exp = - np.log(1 - np.random.rand(10000))  # feeding uniform into inverse cdf

fig, ax = plt.subplots(nrows=1, ncols=2)
# Uniform samples --> Exponential samples
ax[0].plot(x, cdf, label=r'$F(x)=1-e^{-x}$')
ax[0].plot(x, pdf, label=r'$f(x)=e^{-x}$')
ax[0].scatter(exp, np.zeros_like(exp), label='exp samples')
ax[0].hist(exp, bins=50, density=True, label='histogram of samples',
           edgecolor='k', alpha=0.5)
ax[0].set_xlabel(r"$x$")
ax[0].set_ylabel(r"$y$")
ax[0].set_title("Uniform samples into Exponential samples")
ax[0].legend()
# Exponential samples --> Uniform samples
ax[1].hist(1 - np.exp(-exp), bins=50, density=True,
           label='histogram of samples', edgecolor='k')
ax[1].set_title("Exponential Samples into Uniform Samples")
ax[1].axhline(y=1, lw=2, color=orange, xmin=0.045, xmax=0.955,
              label=r"$\mathcal{U}\,(0, 1)$")
ax[1].legend()
ax[1].set_xlabel(r"$x$")
ax[1].set_ylabel(r"$y$")
ax[1].set_ylim(0.0, 1.5)
plt.savefig("images/stochastic_approximations/pit.png")
plt.show()

# Rejection Sampling
fig, ax = plt.subplots(figsize=(10, 8))
fs = 7
x = np.linspace(0, np.pi, 200)
cf = norm(loc=np.pi/2, scale=0.4)
n_samples = 50
cf_samples = cf.rvs(size=n_samples)
uniform_samples = np.random.uniform(np.zeros(n_samples), cf.pdf(cf_samples)*1.7)
ax.plot(x, norm.pdf(x, loc=np.pi/2, scale=0.25), label=r"$\widetilde{p}(x)$")
for i in [1.2, 1.4]:
    ax.plot(x, norm.pdf(x, loc=np.pi/2, scale=0.4) * i, color='k', linestyle='--', alpha=0.1, lw=1)
    mode = norm.pdf(np.pi/2, loc=np.pi/2, scale=0.4) * i
    ax.annotate(r"$k={}$".format(i), [np.pi/2, mode], [1.47, mode+0.03], color=grey, fontsize=fs)
ax.plot(x, norm.pdf(x, loc=np.pi/2, scale=0.4), label=r"$q(x)$")
ax.plot(x, norm.pdf(x, loc=np.pi/2, scale=0.4)*1.7, label=r"$kq(x)$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.annotate(r"$k=1$", [np.pi/2, 1], [1.5, 0.90], color=orange, fontsize=fs)
top = norm.pdf(np.pi/2, loc=np.pi/2, scale=0.4)*1.7
ax.annotate(r"$k=1.7$", [np.pi/2, top], [1.47, top+0.03], color=blue, fontsize=fs)
# in the first plot need to draw samples of the same color
prop = norm(loc=np.pi/2, scale=0.4)
accepted = uniform_samples <= prop.pdf(cf_samples)
ax.scatter(cf_samples[accepted], uniform_samples[accepted], s=7, color=orange,
           zorder=30, label='Accepted Samples', edgecolors='k', linewidth=0.2)
ax.scatter(cf_samples[~accepted], uniform_samples[~accepted], s=7, color=blue,
           zorder=20, label='Rejected Samples', edgecolors='k', linewidth=0.2)
ax.legend()
plt.savefig("images/stochastic_approximations/rejection_sampling.png")
plt.show()
