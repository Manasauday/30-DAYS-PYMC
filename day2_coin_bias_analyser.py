import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def run_coin_model(heads, tails):
    data = [1] * heads + [0] * tails

    with pm.Model() as model:
        p = pm.Beta("p", alpha=1, beta=1)
        obs = pm.Bernoulli("obs", p=p, observed=data)

        trace = pm.sample(1000, cores=1)

    az.plot_posterior(trace)
    plt.show()


if __name__ == "__main__":
    print("Coin Bias Analyzer")
    heads = int(input("Enter number of heads: "))
    tails = int(input("Enter number of tails: "))

    run_coin_model(heads, tails)