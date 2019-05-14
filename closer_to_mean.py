import numpy as np
import matplotlib.pyplot as plt
from no_explanatory_variables import NoExplanatoryVariables
from utility_functions import *


b = 1.0
# laplace means
lap_means = []
# variational means
var_means = []

ns = list(range(30, 100, 5))

for n in ns:
    var_mean = []
    lap_mean = []
    for i in range(50):  # repeat
        # Create model
        seed = np.random.randint(0, 100)
        print(seed)
        y = np.random.binomial(n=1, p=sigmoid(b), size=n)
        dict = {
            'seed': seed,
            'n': n,
            'ybar': np.mean(y),
            'beta': b
        }

        model = NoExplanatoryVariables(save=False, dict=dict)
        # Obtain various means and stuff
        vmean, _ = model._variational_em()
        var_mean.append(abs(vmean - 1.0))
        lap_mean.append(abs(model.sample_mode - 1.0))
    var_means.append(np.mean(var_mean))
    lap_means.append(np.mean(lap_mean))

plt.plot(ns, var_means, label='var_means')
plt.plot(ns, lap_means, label='lap_means')
plt.legend()
plt.show()