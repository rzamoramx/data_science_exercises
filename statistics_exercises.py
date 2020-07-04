""" Some exercises about statistics """
from matplotlib import pyplot as plt
from statistics.central_tendencies import *
from statistics.variance import variance, standard_deviation
from statistics.correlation import covariance, correlation


def main():
    num_friends = [500, 50, 25, 30, 5, 6, 7, 8, 9, 10,
                   1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   19, 28, 37, 46, 55, 64, 73, 82, 91, 10,
                   19, 28, 37, 33, 55, 64, 73, 82, 91, 10]

    daily_minutes = [1, 6, 10, 20, 4, 9, 12, 8, 9, 20,
                     5, 6, 10, 20, 4, 9, 12, 8, 9, 20,
                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    central_tendencies(num_friends)
    dispersion(num_friends)
    correlations(num_friends, daily_minutes)

    correlation_outliers(num_friends, daily_minutes)

    plot_graphs()


def correlation_outliers(num_friends: List[float], daily_minutes: List[float]):
    outlier = num_friends.index(500)

    num_friends_good = [x for i, x in enumerate(num_friends) if i != outlier]
    daily_minutes_good = [x for i, x in enumerate(daily_minutes) if i != outlier]

    # plotting
    plt.figure()
    plt.scatter(num_friends, daily_minutes)
    plt.title("Correlation without outlier")
    plt.xlabel("# of friends")
    plt.ylabel("minutes")

    plt.figure()
    plt.scatter(num_friends_good, daily_minutes_good)
    plt.title("Correlation with outlier")
    plt.xlabel("# of friends")
    plt.ylabel("minutes")


def correlations(num_frieds: List[float], daily_minutes: List[float]):
    cov = covariance(num_frieds, daily_minutes)
    print(f'covariance: {cov}')

    corr = correlation(num_frieds, daily_minutes)
    print(f'correlation: {corr}')


def dispersion(num_friends: List[float]):
    print(data_range(num_friends))

    varian = variance(num_friends)
    print(f'variance: {varian}')

    standard_devi = standard_deviation(num_friends)
    print(f'standard deviation: {standard_devi}')


def central_tendencies(num_friends: List[float]):
    assert median([1, 10, 2, 9, 5]) == 5

    vector_a = [1, 9, 2, 10]
    assert median(vector_a) == (2 + 9) / 2
    print(median(vector_a))

    print(4//2)  # 2
    print(9//2)  # 4

    result_q1 = quantile(num_friends, 0.10)
    print(f'quatile 10%: {result_q1}')

    result_q2 = quantile(num_friends, 0.25)
    print(f'quatile 25%: {result_q2}')

    result_q3 = quantile(num_friends, 0.50)
    print(f'quatile 50%: {result_q3}')

    result_q4 = quantile(num_friends, 0.75)
    print(f'quatile 75%: {result_q4}')

    result_q5 = quantile(num_friends, 0.90)
    print(f'quatile 90%: {result_q5}')

    moda = set(mode(num_friends))
    print(f'moda: {moda}')


def plot_graphs():
    plt.show()


if __name__ == "__main__":
    main()
