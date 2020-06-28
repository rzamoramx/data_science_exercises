from matplotlib import pyplot as plt
from typing import List
import math

Vector = List[float]


def main():
    comp_vectors()
    add_subs_vectors()
    compute_distance()

    plot_graphs()


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


# Base functions for computations in above functions


def distance(v: Vector, w: Vector) -> float:
    """ Computes the distance between v and w by root the result of squared_distance """

    # dist = math.sqrt(squared_distance(v, w))
    # The same result
    return magnitude(subtract(v, w))


def squared_distance(v: Vector, w: Vector) -> float:
    """ Computes (v_1 - w_1)**2 + ... + (v_n - w_n)**2 """
    """ (v_1 - w_1) is compute by subtract that return an vector """
    """ **2 is compute by sum_of_squares that return a float with the squared subtract """
    return sum_of_squares(subtract(v, w))


def magnitude(v: Vector) -> float:
    """ Returns the magnitude (or length) of vector v"""
    return math.sqrt(sum_of_squares(v))


def sum_of_squares(v: Vector) -> float:
    """Uses dot_product function"""
    return dot_product(v, v)


def dot_product(v: Vector, w: Vector) -> float:
    """ Computes v_1 * w_1 + v_2 * w_2 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def vector_mean(vectors: List[Vector]) -> Vector:
    """ Computes the element-wise average"""
    len_vector = len(vectors)

    return scalar_multiply(1/len_vector, vector_sum(vectors))


def scalar_multiply(scalar: float, v: Vector) -> Vector:
    """ Multiplies every element by scalar"""
    assert scalar>0, "Scalar must be greater than zero"

    return [scalar * v_i for v_i in v]


def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    assert vectors, "vectors not provided"

    # First element has the same size that the rest
    num_elements = len(vectors[0])
    # Assert the same size for all elements
    assert all(len(v) == num_elements for v in vectors), "different sizes"

    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


def add(v: Vector, w: Vector) -> Vector:
    """ Adds corresponding elements """
    assert len(v) == len(w), "Vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]


def subtract(v: Vector, w: Vector) -> Vector:
    """ Subtracts corresponding elements"""
    assert len(v) == len(w), "Vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]


if __name__ == "__main__":
    main()
