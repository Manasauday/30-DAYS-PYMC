import numpy as np
import matplotlib.pyplot as plt

def main():
    # simulate 1000 coin tosses
    tosses = np.random.choice([0, 1], size=1000)

    # calculate probability of heads
    probability_heads = np.mean(tosses)

    print("Estimated Probability of Heads:", probability_heads)

    # cumulative probability
    cumulative = np.cumsum(tosses) / np.arange(1, 1001)

    plt.plot(cumulative)
    plt.axhline(0.5)
    plt.title("Convergence of Probability to 0.5")
    plt.xlabel("Number of Tosses")
    plt.ylabel("Probability of Heads")
    plt.show()


if __name__ == "__main__":
    main()