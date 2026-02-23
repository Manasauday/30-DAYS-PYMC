import numpy as np
import matplotlib.pyplot as plt

# true probability (unknown to model)
true_p = 0.7

# generate observations
data = np.random.binomial(1, true_p, size=100)

# estimate probability from data
estimated_p = np.mean(data)

print("Estimated Probability:", estimated_p)

# visualize results
plt.hist(data, bins=2)
plt.title("Observed Coin Toss Results")
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.show()