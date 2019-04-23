import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# Code for Gaussian Lower Bound on sigmoid
# Define sigmoid function, lambda(xi) function and the gaussian lower bound
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lamb(xi):
    return (1 / (2*xi))*(sigmoid(xi) - 0.5)

def gaussian_lb(x, xi):
    return sigmoid(xi)*np.exp((x-xi)/2 - lamb(xi)*(x**2 - xi**2))

# Choose a value for xi. This will indicate the x value where
# Gaussian will be tangent to sigmoid
xi = 2.5
# Choose axes limits
xlim_left = -6
xlim_right = 6
ylim_up = 1
ylim_down = 0

# Plot sigmoid and gaussian between x=-6 and x=6
x = np.linspace(-6, 6, 100)
plt.plot(x, sigmoid(x), label=r'$\sigma(x)$')
plt.plot(x, gaussian_lb(x, xi), label='Gaussian LB', alpha = 0.5)
# Plot xi and a helper line showing that Gaussian LB is tangent to
# sigmoid at x=xi
plt.plot((xi, xi), (0, sigmoid(xi)), linestyle='--', color='k', lw=1)
# improve appearance
plt.xlim(xlim_left, xlim_right)
plt.ylim(ylim_down, ylim_up)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.xticks([xlim_left, -xi, 0, xi, xlim_right], [xlim_left, r'$-\xi$', 0, r'$\xi$', xlim_right])
plt.yticks([ylim_down, 0.5, ylim_up])
plt.title("Gaussian Lower Bound on Sigmoid")
plt.legend()
plt.savefig("images/deterministic_approximations/sigmoid_bound_gaussian.png")
plt.show()