import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Sales data
last_month_sales = 120
this_month_sales = 150

# Calculate observed growth rate
observed_growth = (this_month_sales - last_month_sales) / last_month_sales

with pm.Model() as sales_model:
    
    # Prior belief about growth rate (normal distribution)
    growth_rate = pm.Normal("growth_rate", mu=0.1, sigma=0.2)
    
    # Likelihood (observed growth with small noise)
    likelihood = pm.Normal(
        "observed_growth",
        mu=growth_rate,
        sigma=0.05,
        observed=observed_growth
    )
    
    # Sampling
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=2,
        cores=1,
        random_seed=42,
        progressbar=True
    )

# Posterior mean
posterior_samples = trace.posterior["growth_rate"].values.flatten()
mean_growth = np.mean(posterior_samples)

print("Estimated Monthly Sales Growth Rate:", round(mean_growth, 3))

# Plot posterior
pm.plot_posterior(trace, var_names=["growth_rate"])
plt.show()