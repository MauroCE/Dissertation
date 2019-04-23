# Import external libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set parameters to make plots look more professional
sns.set_style('darkgrid')
plt.rcParams['mathtext.fontset'] =  'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 6)
plt.rc('figure', titlesize=14)

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
ax[1].axhline(y=1, lw=2, color="#ff7f0e", xmin=0.045, xmax=0.955,
              label=r"$\mathcal{U}\,(0, 1)$")
ax[1].legend()
ax[1].set_xlabel(r"$x$")
ax[1].set_ylabel(r"$y$")
ax[1].set_ylim(0.0, 1.5)
plt.show()
