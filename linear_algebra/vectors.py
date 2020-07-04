""" Base functions for computations in above functions """
from typing import List
import math

Vector = List[float]


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
    """ Sums all corresponding elements """
    assert vectors, "vectors not provided"

    # First element has the same size that the rest
    num_elements = len(vectors[0])
    # Assert the same size for all elements
    assert all(len(v) == num_elements for v in vectors), "different sizes"

    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


def add(v: Vector, w: Vector) -> Vector:
    """ Adds corresponding elements on v and w one to one """
    assert len(v) == len(w), "Vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]


def subtract(v: Vector, w: Vector) -> Vector:
    """ Subtracts corresponding elements on v and w one to one """
    assert len(v) == len(w), "Vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]
