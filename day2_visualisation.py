import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt


def run_model():
    data = np.random.binomial(1, 0.7, size=100)

    with pm.Model() as model:
        p = pm.Beta("p", alpha=1, beta=1)
        obs = pm.Bernoulli("obs", p=p, observed=data)

        trace = pm.sample(1000, cores=1)

    az.plot_posterior(trace)
    plt.show()


if __name__ == "__main__":
    run_model()