""" Base functions for computes variances """
import math
from typing import List
from linear_algebra.vectors import sum_of_squares
from statistics.central_tendencies import de_mean


def standard_deviation(xs: List[float]) -> float:
    """ The standard deviation is the square root of the variance """
    return math.sqrt(variance(xs))


def variance(xs: List[float]) -> float:
    """ Almost the average squared deviation from the mean """
    assert len(xs) >= 2, "variance requires at least two elements"

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)