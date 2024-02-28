import numpy as np


POSITIONS = [
    [2.0, 0.0, 1.0],
    [1.08, 1.68, 2.38],
    [-0.83, 1.82, 2.49],
    [-1.97, 0.28, 2.15],
    [-1.31, -1.51, 2.59],
    [0.57, -1.91, 4.32],
]


def sum_of_square_error(y_s, x_s, polynomial_func):
    predictions = [polynomial_func(x) for x in x_s]
    return np.sum((y_s - predictions) ** 2)
