import numpy as np
import matplotlib.pyplot as plt
from no_explanatory_variables import NoExplanatoryVariables
from utility_functions import sigmoid

n_iter = 100
steps = 1000
x = np.linspace(-10, 10, steps)
tvs = np.zeros((n_iter, steps))
tls = np.zeros((n_iter, steps))


for i in range(n_iter):
    n = 1000
    theta = 1
    y = np.random.binomial(n=1, p=sigmoid(theta), size=n)
    mydict = {
        'seed': np.random.randint(0, 10000),
        'n': n,
        'ybar': np.mean(y)
    }
    np.random.seed(mydict['seed'])
    model = NoExplanatoryVariables(False, dict=mydict)
    # _ = model.sample(s=100000, b=500, t=1, scale=0.25, kde_scale=0.15)
    tvs[i, :] = abs(model.true_log_posterior(x) - model.log_variational(x))
    tls[i, :] = abs(model.true_log_posterior(x) - model.log_laplace(x))


fig, ax = plt.subplots()
tv_mean = tvs.mean(axis=0)
tl_mean = tls.mean(axis=0)
ax.plot(x, tv_mean, label='abs(true-var)')
ax.plot(x, tl_mean, label='abs(true-lap)')
ax.legend()
plt.show()
