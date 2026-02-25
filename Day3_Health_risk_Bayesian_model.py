import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Observed health data
patients_tested = 100
disease_cases = 18

# Bayesian Model
with pm.Model() as health_model:

    # Prior belief: disease probability between 0 and 1
    disease_rate = pm.Beta("disease_rate", alpha=2, beta=8)

    # Likelihood: observed cases from tested patients
    observed_cases = pm.Binomial(
        "observed_cases",
        n=patients_tested,
        p=disease_rate,
        observed=disease_cases
    )

    # Sampling
    trace = pm.sample(
        2000,
        tune=1000,
        cores=1,
        random_seed=42,
        progressbar=True
    )

# Summary results
print(az.summary(trace, var_names=["disease_rate"]))

# Posterior plot
az.plot_posterior(trace, var_names=["disease_rate"])
plt.show()