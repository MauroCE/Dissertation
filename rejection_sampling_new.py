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
#np.random.seed(4)
np.random.seed(5)

# Rejection Sampling
fig, ax = plt.subplots(figsize=(10, 8))
fs = 7
x = np.linspace(0, np.pi, 200)
cf = norm(loc=np.pi/2, scale=0.4)
n_samples = 50
cf_samples = cf.rvs(size=n_samples)
uniform_samples = np.random.uniform(np.zeros(n_samples), cf.pdf(cf_samples)*1.7)
ax.plot(x, norm.pdf(x, loc=np.pi/2, scale=0.25), label=r"$p(x\mid \theta)p(\theta)$")
for i in [1.2, 1.4]:
    ax.plot(x, norm.pdf(x, loc=np.pi/2, scale=0.4) * i, color='k', linestyle='--', alpha=0.1, lw=1)
    mode = norm.pdf(np.pi/2, loc=np.pi/2, scale=0.4) * i
    ax.annotate(r"$k={}$".format(i), [np.pi/2, mode], [1.52, mode+0.03], color=grey, fontsize=fs)
ax.plot(x, norm.pdf(x, loc=np.pi/2, scale=0.4), color='k', linestyle='--', alpha=0.1, lw=1) # label=r"$q(\theta)$"
ax.plot(x, norm.pdf(x, loc=np.pi/2, scale=0.4)*1.7, label=r"$kq(\theta)$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
modek1 = norm.pdf(np.pi/2, loc=np.pi/2, scale=0.4)
ax.annotate(r"$k=1$", [np.pi/2, 1], [1.52, modek1 + 0.03], color=grey, fontsize=fs)
top = norm.pdf(np.pi/2, loc=np.pi/2, scale=0.4)*1.7
ax.annotate(r"$k=1.7$", [np.pi/2, top], [1.53, top+0.03], color=orange, fontsize=fs)
# in the first plot need to draw samples of the same color
prop = norm(loc=np.pi/2, scale=0.4)
accepted = uniform_samples <= norm.pdf(cf_samples, loc=np.pi/2, scale=0.25) #prop.pdf(cf_samples)
ax.scatter(cf_samples[accepted], uniform_samples[accepted], s=7, color='#1f77b4',
           zorder=30, label='Accepted Samples', edgecolors='k', linewidth=0.2)
ax.scatter(cf_samples[~accepted], uniform_samples[~accepted], s=7, color='#ff7f0e',
           zorder=20, label='Rejected Samples', edgecolors='k', linewidth=0.2)
ax.legend()
plt.savefig("images/stochastic_approximations/rejection_sampling_new.png")
plt.show()