import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# Observed recovery times (days)
recovery_days = np.array([5, 7, 6, 8, 7, 6, 9, 5, 6, 7, 8, 6])

with pm.Model() as recovery_model:

    # Prior for mean recovery time
    mu = pm.Normal("mu", mu=7, sigma=3)

    # Prior for standard deviation
    sigma = pm.HalfNormal("sigma", sigma=2)

    # Likelihood
    observed = pm.Normal(
        "observed",
        mu=mu,
        sigma=sigma,
        observed=recovery_days
    )

    # Sampling
    trace = pm.sample(
        2000,
        tune=1000,
        cores=1,
        random_seed=42,
        progressbar=True
    )

# Summary
print(az.summary(trace, var_names=["mu", "sigma"]))

# Posterior plots
az.plot_posterior(trace, var_names=["mu", "sigma"])
plt.show()