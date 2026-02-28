import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

np.random.seed(42)

n = 300
true_vol = np.exp(np.random.normal(-3, 0.3, n))
returns = np.random.normal(0, true_vol)

with pm.Model() as model:
    sigma = pm.Exponential("sigma", 50)
    volatility = pm.GaussianRandomWalk("volatility", sigma=sigma, shape=n)
    returns_obs = pm.Normal("returns_obs", mu=0, sigma=pm.math.exp(volatility), observed=returns)

    trace = pm.sample(
        draws=2000,
        tune=1000,
        target_accept=0.9,
        return_inferencedata=True
    )

print(az.summary(trace, var_names=["sigma"], round_to=4))

posterior_vol = trace.posterior["volatility"].mean(dim=("chain", "draw")).values

plt.plot(np.exp(posterior_vol))
plt.title("Estimated Time-Varying Volatility")
plt.show()

print("Stochastic Volatility Model Ran Successfully 🚀")