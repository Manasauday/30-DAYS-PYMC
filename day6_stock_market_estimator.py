# Bayesian Stock Return & Volatility Model
# Works with PyMC 5+

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# -----------------------------
# 1. Simulate Stock Returns
# -----------------------------
np.random.seed(42)

n_days = 252  # 1 trading year
true_mu = 0.001      # average daily return
true_sigma = 0.02    # daily volatility

returns = np.random.normal(true_mu, true_sigma, n_days)

# Create price series
price = 100 * np.exp(np.cumsum(returns))

# -----------------------------
# 2. Bayesian Model
# -----------------------------
with pm.Model() as model:

    # Priors
    mu = pm.Normal("mu", mu=0, sigma=0.01)
    sigma = pm.HalfNormal("sigma", sigma=0.05)

    # Likelihood
    likelihood = pm.Normal(
        "returns",
        mu=mu,
        sigma=sigma,
        observed=returns
    )

    # Sample
    trace = pm.sample(
        draws=2000,
        tune=1000,
        target_accept=0.9,
        return_inferencedata=True
    )

# -----------------------------
# 3. Results
# -----------------------------
print(az.summary(trace, round_to=4))

az.plot_trace(trace)
plt.show()

# -----------------------------
# 4. Forecast Future Prices
# -----------------------------
posterior_mu = trace.posterior["mu"].mean().item()
posterior_sigma = trace.posterior["sigma"].mean().item()

future_days = 30
future_returns = np.random.normal(posterior_mu, posterior_sigma, future_days)

future_price = price[-1] * np.exp(np.cumsum(future_returns))

plt.plot(price, label="Historical Price")
plt.plot(
    range(len(price), len(price) + future_days),
    future_price,
    linestyle="--",
    label="Forecast"
)
plt.legend()
plt.title("Bayesian Stock Price Forecast")
plt.show()

print("\nModel ran successfully 🚀")