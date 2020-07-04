""" Base functions for computes central tendencies """
from collections import Counter
from typing import List

Vector = List[float]


def de_mean(xs: List[float]) -> List[float]:
    """ Translate xs by subtracting its mean (so the result has mean 0) """
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)


def mode(x: List[float]) -> List[float]:
    """ Returns a list, since there might be more than one mode """
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]


def quantile(xs: List[float], p: float) -> float:
    """ Returns the pth-percentile value in x """
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


def _median_odd(xs: Vector) -> float:
    """ If len(xs) is odd(impar), the median is the middle element """
    return sorted(xs)[len(xs) // 2]


def _median_even(xs: Vector) -> float:
    """ If len(xs) is even(entero), it's the avg of the middle two elements """
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2  # 4 => 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2


def median(v: Vector) -> float:
    """ Finds the middle most value appear in vector """
    return _median_even(v) if len(v)%2 == 0 else _median_odd(v)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)
