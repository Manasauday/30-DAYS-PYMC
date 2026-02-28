import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

np.random.seed(42)

n_days = 300
true_mu = 0.0005
true_sigma = 0.02

returns = np.random.normal(true_mu, true_sigma, n_days)
crash_threshold = -0.05

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=0.01)
    sigma = pm.HalfNormal("sigma", sigma=0.05)

    pm.Normal("returns", mu=mu, sigma=sigma, observed=returns)

    trace = pm.sample(
        draws=2000,
        tune=1000,
        target_accept=0.9,
        return_inferencedata=True
    )

posterior_mu = trace.posterior["mu"].values.flatten()
posterior_sigma = trace.posterior["sigma"].values.flatten()

posterior_samples = np.random.normal(
    posterior_mu,
    posterior_sigma
)

crash_probability = np.mean(posterior_samples < crash_threshold)

print(az.summary(trace, round_to=4))
print("\nCrash Probability (Return < -5%):")
print(f"{crash_probability * 100:.2f}%")

plt.hist(posterior_samples, bins=100)
plt.axvline(crash_threshold)
plt.title("Posterior Predictive Return Distribution")
plt.show()