import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

# Parameters for prettier plots
sns.set_style('darkgrid')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 12

np.random.seed(39)
color1 = 'k'
color2 = '#DD6C21'


def normal_unnormalized(x):
    return np.exp(-x**2 / 2)

f = -4
t = 4

def approx_norm_const(N_list, f=f, t=t):
    all_results = []
    N_max = np.max(N_list)
    u = np.random.uniform(low=f, high=t, size=N_max)
    for N in N_list:
        # dictionary to store everything
        results = {'u': u[:N]}
        # WE PROBABLY NEED TO MULTIPLY BY SIZE OF THE INTERVAL THAT WE are sampling from
        size_interval = t - f
        # evaluate samples with unnormalized normal
        results['unnormalized'] = normal_unnormalized(results['u'])
        # sum them up to find normalizing constant
        results['norm_const'] = (np.sum(results['unnormalized']) / N) * size_interval
        # sort the uniform random samples to plot them
        results['sorted_samples'] = np.sort(results['u'])
        # find sorted evaluations for plotting
        results['sorted_evaluations'] = normal_unnormalized(results['sorted_samples']) / results['norm_const']
        # store number for convenience
        results['N'] = N
        all_results.append(results)
    return all_results


# Normal random variable
x = np.linspace(f, t, 500)
y = norm.pdf(x)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, y, label=r'$\mathcal{N}(0, 1)$' + ', {:.4}'.format(np.sqrt(2*np.pi)), lw=1, color=color1, zorder=10)

results = approx_norm_const([50, 500, 5000])
labels = []
line_styles = ['-.', ':', '--']

for index, res in enumerate(results):
    ax.plot(res['sorted_samples'], res['sorted_evaluations'],
            lw=1, alpha=1, color=color2, ls=line_styles[index])
    labels.append('# Samples={}, Normalization Constant'.format(res['N']) + r'$\approx$'+'{:.3}'.format(res['norm_const']))



lines = ax.get_lines()
legend1 = ax.legend([lines[0]],
                    [r'$\mathcal{N}(0, 1)$' + ', Normalizing Constant'+r'$\approx$'+'{:.3}'.format(np.sqrt(2*np.pi))],
                    loc=1)
legend2 = ax.legend([lines[i] for i in range(1, len(results)+1)], labels, loc=2)
ax.add_artist(legend1)
ax.add_artist(legend2)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$p(\theta \mid x)$')
plt.savefig("images/stochastic_approximations/uniform_mc.png")
plt.show()
