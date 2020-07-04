""" Some exercises about linear algebra """
from matplotlib import pyplot as plt
from linear_algebra.vectors import *
from linear_algebra.matrix import get_row, get_column, shape


def main():
    add_subs_vectors()
    comp_vectors()
    compute_distance()

    plot_graphs()

    basic_operations_on_matrix()


def basic_operations_on_matrix():
    matrix_a = [[1, 2, 3], [4, 6, 3]]
    assert shape(matrix_a) == (2, 3)

    print(matrix_a)

    print(get_row(matrix_a, 1))
    print(get_column(matrix_a, 1))


def plot_graphs():
    plt.show()


def comp_vectors():
    assert vector_sum([[3, 2], [5, 6], [2, 1]]) == [10, 9]
    assert scalar_multiply(2, [2, 4, 6]) == [4, 8, 12]

    assert vector_mean([[2, 3], [4, 5], [6, 7]]) == [4, 5]
    assert vector_mean([[2, 3, 4], [3, 4, 5], [4, 5, 6]]) == [3, 4, 5]

    assert dot_product([1, 2, 3], [4, 5, 6]) == 32

    assert sum_of_squares([1, 2, 3]) == 14  # 1*1 + 2*2 + 3*3

    assert magnitude([3, 4]) == 5


def add_subs_vectors():
    vector_a = [1, 2, 3]
    vector_b = [3, 2, 1]

    vector_sums = add(vector_a, vector_b)
    vector_subs = subtract(vector_a, vector_b)

    assert add(vector_a, vector_b) == [4, 4, 4]
    assert subtract(vector_a, vector_b) == [-2, 0, 2]

    x_label = [1, 2, 3]

    plt.figure()
    plt.plot(x_label, vector_a, color='blue', marker='o', linestyle='solid')
    plt.plot(x_label, vector_b, color='green', marker='o', linestyle='solid')
    plt.plot(x_label, vector_sums, color='red', marker='o', linestyle='solid')
    plt.plot(x_label, vector_subs, color='black', marker='o', linestyle='dotted')
    plt.title("Vectors")

    print(vector_a)
    print(vector_b)
    print(vector_sum)
    print(vector_subs)


def compute_distance():
    vector_1 = [2, 3]
    vector_2 = [4, 5]
    dist = distance(vector_1, vector_2)
    print(f'distance of {vector_1} and {vector_2} is: {dist}')

    xs, ys = [pair for pair in zip(vector_1, vector_2)]
    print(xs)
    print(ys)

    plt.figure()
    plt.scatter(xs, ys)


if __name__ == "__main__":
    main()