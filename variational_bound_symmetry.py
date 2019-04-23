import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set_style('darkgrid')
plt.rcParams['figure.dpi']= 300
plt.rc("savefig", dpi=300)
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lambda_func(x):
    s = sigmoid(x)
    return (s - 0.5) / (2*x)

def contour_plot(func, xleft, xright, yleft, yright, num, title="", xlabel="", ylabel=""):
    """
    Given a R2 -> R function "func" which takes an array of dimension 2 and returns a scalar,
    this function will plot a surface plot over the specified region.
    """
    b1 = np.linspace(xleft, xright, num)
    b2 = np.linspace(yleft, yright, num)
    B1, B2 = np.meshgrid(b1, b2)
    Zbeta = np.array([func(beta) for beta in zip(B1.ravel(), B2.ravel())])
    Zbeta = Zbeta.reshape(B1.shape)
    fig, ax = plt.subplots()
    ax.contour(B1, B2, Zbeta)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax

def bound_2d(both):
    xi, z = both
    return sigmoid(xi)*np.exp((z-xi)/2-lambda_func(xi)*(z**2-xi**2))

ax = contour_plot(bound_2d, -10, 10,-10, 10, 100)
ax.set_xlabel(r'$\xi_i$')
ax.set_ylabel(r'$\mathbf{x}_i^\top\mathbf{\beta}$')
ax.set_title(r"Contour of bound as both $\xi_i$ and $\mathbf{x}_i^\top\mathbf{\beta}$ change.")
ax.axvline(x=0, lw=1, color='b', ls='--', alpha=0.6)
plt.savefig('images/deterministic_approximations/contour_bound.png')
plt.show()