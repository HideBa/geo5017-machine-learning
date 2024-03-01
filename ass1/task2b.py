import math
import numpy as np
from plot import plot_xyz
from common import POSITIONS, sum_of_square_error


def second_degree_polynomial(a0, a1, a2, x):
    return a0 + a1 * x + a2 * (x**2)


def high_order_gradient(x_s, y_s):

    def gradient_func(a0, a1, a2):
        y_pred = [second_degree_polynomial(a0, a1, a2, x) for x in x_s]
        residuals = y_s - y_pred
        # partial derivative of the loss function with respect to a0, a1 and a2
        a0_grad = -2 * np.sum(residuals)
        a1_grad = -2 * np.sum((residuals) * x_s)
        a2_grad = -2 * np.sum((residuals) * (x_s**2))
        return np.array([a0_grad, a1_grad, a2_grad])

    return gradient_func


def gradient_descent(
    start, gradient_func, learning_rate=0.01, iterations=1000, tolerance=0.01
):
    # start is a list of initial values [a0, a1]
    a0_steps, a1_steps, a2_steps = [start[0]], [start[1]], [start[2]]
    a0, a1, a2 = start

    for _ in range(iterations):
        gradient = gradient_func(a0, a1, a2)
        diff = math.sqrt(
            gradient[0] ** 2 + gradient[1] ** 2 + gradient[2] ** 2
        )

        if diff < tolerance:
            break
        a0 -= gradient[0] * learning_rate
        a1 -= gradient[1] * learning_rate
        a2 -= gradient[2] * learning_rate
        a0_steps.append(a0)
        a1_steps.append(a1)
        a2_steps.append(a2)
    return a0_steps, a1_steps, a2_steps, a0, a1, a2


def second_degree_polynomial_regression(x_s, y_s):

    gradient_func = high_order_gradient(x_s, y_s)

    _, _, _, a0, a1, a2 = gradient_descent(
        [0, 0, 0],
        gradient_func,
        learning_rate=0.0001,
        iterations=100000,
        tolerance=0.01,
    )

    sse = sum_of_square_error(y_s, x_s, lambda x: a0 + a1 * x + a2 * x**2)
    return ((a0, a1, a2), sse)


def main():
    time_steps = np.array([i for i in range(len(POSITIONS))])
    x, y, z = zip(*POSITIONS)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    (x_a0, x_a1, x_a2), x_sse = second_degree_polynomial_regression(
        time_steps, x
    )
    (y_a0, y_a1, y_a2), y_sse = second_degree_polynomial_regression(
        time_steps, y
    )
    (z_a0, z_a1, z_a2), z_sse = second_degree_polynomial_regression(
        time_steps, z
    )

    print("SSE(x): ", x_sse)
    print("SSE(y): ", y_sse)
    print("SSE(z): ", z_sse)

    next_position = [
        second_degree_polynomial(x_a0, x_a1, x_a2, time_steps[-1] + 1),
        second_degree_polynomial(y_a0, y_a1, y_a2, time_steps[-1] + 1),
        second_degree_polynomial(z_a0, z_a1, z_a2, time_steps[-1] + 1),
    ]

    x = np.append(x, next_position[0])
    y = np.append(y, next_position[1])
    z = np.append(z, next_position[2])

    # make a list of [x, y, z] for plot_xyz
    positions = np.array(list(zip(x, y, z)))
    plot_xyz(positions, "task2b")


if __name__ == """__main__""":
    main()
