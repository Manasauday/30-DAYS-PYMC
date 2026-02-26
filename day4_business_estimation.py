import pymc as pm
import matplotlib.pyplot as plt

# Observed business data
total_campaigns = 40
successful_campaigns = 15

with pm.Model() as business_model:
    
    # Prior belief about success probability
    success_prob = pm.Beta("success_prob", alpha=2, beta=2)
    
    # Likelihood from observed data
    outcomes = pm.Binomial(
        "outcomes",
        n=total_campaigns,
        p=success_prob,
        observed=successful_campaigns
    )
    
    # Sampling
    trace = pm.sample(
        1000,
        tune=1000,
        chains=2,
        cores=1,
        progressbar=True,
        random_seed=42
    )

# Posterior estimate
mean_success = trace.posterior["success_prob"].values.mean()
print("Estimated business success probability:", round(mean_success, 3))

# Plot posterior distribution
pm.plot_posterior(trace, var_names=["success_prob"])
plt.show()