import math
import numpy as np
from plot import plot_xyz
from common import POSITIONS, sum_of_square_error


def high_order_gradient(x_s, y_s):
    def first_degree_polynomial(a0, a1, x):
        return a0 + a1 * x

    def gradient_func(a0, a1):
        y_pred = [first_degree_polynomial(a0, a1, x) for x in x_s]
        sse = np.sum((y_s - y_pred) ** 2)
        # partial derivative of the loss function with respect to a0 and a1
        a0_grad = -2 * np.sum(y_s - y_pred)
        a1_grad = -2 * np.sum((y_s - y_pred) * x_s)
        return np.array([a0_grad, a1_grad]), sse

    return gradient_func


def gradient_descent(
    start, gradient_func, learning_rate=0.01, iterations=1000, tolerance=0.01
):
    # start is a list of initial values [a0, a1]
    a0_steps, a1_steps, sse_steps = [start[0]], [start[1]], [0]
    a0, a1 = start

    for _ in range(iterations):
        gradient, sse = gradient_func(a0, a1)
        diff = math.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
        if diff < tolerance:
            break
        a0 -= gradient[0] * learning_rate
        a1 -= gradient[1] * learning_rate
        a0_steps.append(a0)
        a1_steps.append(a1)
        sse_steps.append(sse)
    return [a0_steps, a1_steps, sse_steps], [a0, a1]


def first_degree_polynomial_regression(x_s, y_s):

    gradient_func = high_order_gradient(x_s, y_s)

    _, [a0, a1] = gradient_descent(
        [0, 0],
        gradient_func,
        learning_rate=0.01,
        iterations=1000,
        tolerance=0.01,
    )

    # Here we can get two of coefficient and get formula of y = a0 + a1 * x
    # In this assignement, a1 will be speed of drone
    sse = sum_of_square_error(y_s, x_s, lambda x: a0 + a1 * x)
    return ((a0, a1), sse)


def main():
    time_steps = np.array([i for i in range(len(POSITIONS))])
    x, y, z = zip(*POSITIONS)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    (x_a0, x_a1), x_sse = first_degree_polynomial_regression(time_steps, x)
    (y_a0, y_a1), y_sse = first_degree_polynomial_regression(time_steps, y)
    (z_a0, z_a1), z_sse = first_degree_polynomial_regression(time_steps, z)

    print("SSE(x): ", x_sse)
    print("SSE(y): ", y_sse)
    print("SSE(z): ", z_sse)

    next_position = [
        x_a0 + x_a1 * (time_steps[-1] + 1),
        y_a0 + y_a1 * (time_steps[-1] + 1),
        z_a0 + z_a1 * (time_steps[-1] + 1),
    ]

    print("Next position is: ", next_position)

    x = np.append(x, next_position[0])
    y = np.append(y, next_position[1])
    z = np.append(z, next_position[2])

    # make a list of [x, y, z] for plot_xyz
    positions = np.array(list(zip(x, y, z)))
    plot_xyz(positions, "task2a")


if __name__ == """__main__""":
    main()
