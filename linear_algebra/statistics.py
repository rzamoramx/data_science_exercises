from typing import List

Vector = List[float]


def main():
    central_tendencies()


def central_tendencies():
    assert median([1, 10, 2, 9, 5]) == 5

    vector_a = [1, 9, 2, 10]
    assert median(vector_a) == (2 + 9) / 2
    print(median(vector_a))

    print(4//2)  # 2
    print(9//2)  # 4


# Base functions for computes


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


if __name__ == "__main__":
    main()