import pymc as pm
import matplotlib.pyplot as plt

# Observed data
total_patients = 30
positive_cases = 8

with pm.Model() as model:
    
    # Prior: probability of disease (unknown)
    disease_prob = pm.Beta("disease_prob", alpha=2, beta=2)
    
    # Likelihood: observed positives
    cases = pm.Binomial(
        "cases",
        n=total_patients,
        p=disease_prob,
        observed=positive_cases
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

# Posterior mean
mean_prob = trace.posterior["disease_prob"].values.mean()
print("Estimated disease probability:", round(mean_prob, 3))

# Plot
pm.plot_posterior(trace, var_names=["disease_prob"])
plt.show()