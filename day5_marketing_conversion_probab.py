import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Marketing campaign data
total_clicks = 500
conversions = 65

with pm.Model() as marketing_model:
    
    # Prior belief about conversion probability
    conversion_prob = pm.Beta("conversion_prob", alpha=2, beta=8)
    
    # Likelihood from observed data
    observed_data = pm.Binomial(
        "observed_data",
        n=total_clicks,
        p=conversion_prob,
        observed=conversions
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

# Posterior estimate
posterior_samples = trace.posterior["conversion_prob"].values.flatten()
mean_conversion = np.mean(posterior_samples)

print("Estimated Marketing Conversion Probability:", round(mean_conversion, 4))

# Plot posterior distribution
pm.plot_posterior(trace, var_names=["conversion_prob"])
plt.show()