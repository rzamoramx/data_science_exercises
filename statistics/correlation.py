""" Base function for computes correlations """
from typing import List
from linear_algebra.vectors import dot_product
from statistics.central_tendencies import de_mean
from statistics.variance import standard_deviation


def correlation(xs: List[float], ys: List[float]) -> float:
    """ Measures how much xs and ys vary in tandem about their means """
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0 # if no variation, correlation is zero


def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs and ys must have some number of elements"

    return dot_product(de_mean(xs), de_mean(ys)) / (len(xs) - 1)
