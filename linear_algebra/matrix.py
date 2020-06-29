from typing import List, Tuple

Matrix = List[List[float]]
Vector = List[float]


def main():
    basic_operations_on_matrix()


def basic_operations_on_matrix():
    matrix_a = [[1, 2, 3], [4, 6, 3]]
    assert shape(matrix_a) == (2, 3)

    print(matrix_a)

    print(get_row(matrix_a, 1))
    print(get_column(matrix_a, 1))


# Base functions for computes matrix


def get_column(a: Matrix, c: int) -> Vector:
    """ Returns the c element column of a as Vector """
    return [a_i[c]  # the columns of a_i
            for a_i in a]  # for each row


def get_row(a: Matrix, i: int) -> Vector:
    """ Returns the i element row of a as a Vector """
    return a[i]


def shape(a: Matrix) -> Tuple[int, int]:
    """ Returns the num of rows and columns of A """
    num_rows = len(a)
    num_cols = len(a[0]) if a else 0  # number of elements so columns in first element if it exists
    return num_rows, num_cols

if __name__ == "__main__":
    main()