import pymc as pm
import matplotlib.pyplot as plt
import numpy as np

# Observed startup data
total_startups = 60
successful_startups = 18

with pm.Model() as startup_model:
    
    # Prior belief about startup success probability
    success_prob = pm.Beta("success_prob", alpha=2, beta=2)
    
    # Likelihood based on observed data
    observed_success = pm.Binomial(
        "observed_success",
        n=total_startups,
        p=success_prob,
        observed=successful_startups
    )
    
    # Sampling from posterior
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=2,
        cores=1,
        random_seed=42,
        progressbar=True
    )

# Posterior mean estimate
posterior_samples = trace.posterior["success_prob"].values.flatten()
mean_estimate = np.mean(posterior_samples)

print("Estimated Startup Success Probability:", round(mean_estimate, 3))

# Plot posterior distribution
pm.plot_posterior(trace, var_names=["success_prob"])
plt.show()