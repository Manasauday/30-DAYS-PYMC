print("HELLO ! DAY 1 PYMC STREAK STARTED")
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.random.normal(0, 1, 100)

    mean = np.mean(data)
    std = np.std(data)

    print("Mean:", mean)
    print("Standard Deviation:", std)

    # plot
    plt.hist(data, bins=20)
    plt.title("Random Data Distribution")
    plt.show()


if __name__ == "__main__":
    main()