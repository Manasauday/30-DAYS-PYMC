import pymc as pm
import matplotlib.pyplot as plt

# Observed cricket data
balls_faced = 50
scoring_shots = 18   # balls where player scored runs

with pm.Model() as cricket_model:
    
    # Prior belief about player's scoring probability
    scoring_prob = pm.Beta("scoring_prob", alpha=2, beta=2)
    
    # Likelihood from observed data
    shots = pm.Binomial(
        "shots",
        n=balls_faced,
        p=scoring_prob,
        observed=scoring_shots
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
mean_prob = trace.posterior["scoring_prob"].values.mean()
print("Estimated player scoring probability:", round(mean_prob, 3))

# Plot result
pm.plot_posterior(trace, var_names=["scoring_prob"])
plt.show()