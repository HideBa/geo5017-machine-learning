import plotly.graph_objects as go
import numpy as np
import math


""" Q1:independent variable, dependent variable, assumption that their relationship is linear, a prior knowledge of the dataset

    Q2: The optimal solution is to find the best fit line that minimizes the error. The
    method of finding the minimal error is known as the ordinary least squares method."""

x = [2, 1.08, -0.83, -1.97, -1.31, 0.57]
y = [0, 1.68, 1.82, 0.28, -1.51, -1.91]
z = [1, 2.38, 2.49, 2.15, 2.59, 4.32]
t = [0, 1, 2, 3, 4, 5]

def trajectory():
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=5))])

    fig.update_layout(title='Trajectory of Tracked Positions', xaxis_title='X', yaxis_title='Y',
                      scene=dict(zaxis=dict(title='Z')))
    return fig.show()


def speed_slr(x,y,z,t):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    t = np.array(t)

    xa2 = np.cov(t,x)[0,1]/np.var(t, ddof=1)
    xa1 = np.mean(x) - xa2 * np.mean(t)

    yb2 = np.cov(t, y)[0,1] / np.var(t, ddof=1)
    yb1 = np.mean(y) - yb2 * np.mean(t)

    zc2 = np.cov(t, z)[0,1] / np.var(t, ddof=1)
    zc1 = np.mean(z) - zc2 * np.mean(t)

    x_speed_eq = lambda time: xa1 + xa2 * time
    y_speed_eq = lambda time: yb1 + yb2 * time
    z_speed_eq = lambda time: zc1 + zc2 * time

    # x_residuals = x - x_speed_eq(t)
    # y_residuals = y - y_speed_eq(t)
    # z_residuals = z - z_speed_eq(t)
    #
    # x_residuals_squared_sum = np.sum(x_residuals ** 2)
    # y_residuals_squared_sum = np.sum(y_residuals ** 2)
    # z_residuals_squared_sum = np.sum(z_residuals ** 2)



    print(f"x_speed_eq(t) = {xa1:.2f} + {xa2:.2f}t")
    print(f"y_speed_eq(t) = {yb1:.2f} + {yb2:.2f}t")
    print(f"z_speed_eq(t) = {zc1:.2f} + {zc2:.2f}t")

    return x_speed_eq, y_speed_eq, z_speed_eq


def acc_pr(x,y,z,t):
    coefficients = np.polyfit(t, x, 2)
    len_t = len(t)
    sum_t = np.sum(t)
    sum_t2 = sum(t_i * t_i for t_i, t_i in zip(t, t))
    sum_t3 = sum(t_i * t_i * t_i for t_i, t_i, t_i in zip(t, t, t))
    sum_t4 = sum(t_i * t_i * t_i * t_i for t_i, t_i, t_i, t_i in zip(t, t, t, t))

    sum_x = np.sum(x)
    sum_xt = sum(x_i * t_i for x_i, t_i in zip(x, t))
    sum_xt2 = sum(t_i * t_i * x_i for t_i, t_i, x_i in zip(t, t, x))

    sum_y = np.sum(y)
    sum_yt = sum(y_i * t_i for y_i, t_i in zip(y, t))
    sum_yt2 = sum(t_i * t_i * y_i for t_i, t_i, y_i in zip(t, t, y))

    sum_z = np.sum(z)
    sum_zt = sum(z_i * t_i for z_i, t_i in zip(z, t))
    sum_zt2 = sum(t_i * t_i * z_i for t_i, t_i, z_i in zip(t, t, z))

    xa1 = np.array([[len_t, sum_t, sum_t2],[sum_t, sum_t2, sum_t3], [sum_t2, sum_t3, sum_t4]])
    xam = np.array([sum_x, sum_xt, sum_xt2])
    xa = np.linalg.solve(xa1,xam)

    yb1 = np.array([[len_t, sum_t, sum_t2], [sum_t, sum_t2, sum_t3], [sum_t2, sum_t3, sum_t4]])
    ybm = np.array([sum_y, sum_yt, sum_yt2])
    yb = np.linalg.solve(yb1, ybm)

    zc1 = np.array([[len_t, sum_t, sum_t2], [sum_t, sum_t2, sum_t3], [sum_t2, sum_t3, sum_t4]])
    zcm = np.array([sum_z, sum_zt, sum_zt2])
    zc = np.linalg.solve(zc1, zcm)

    x_acc_eq = lambda t_val: xa[0] + xa[1] * t_val + xa[2] * t_val ** 2
    y_acc_eq = lambda t_val: yb[0] + yb[1] * t_val + yb[2] * t_val ** 2
    z_acc_eq = lambda t_val: zc[0] + zc[1] * t_val + zc[2] * t_val ** 2

    print(f"x_acc_eq(t) = {xa[0]:.4f} + {xa[1]:.4f}t + {xa[2]:.4f}t^2")
    print(f"y_acc_eq(t) = {yb[0]:.4f} + {yb[1]:.4f}t + {yb[2]:.4f}t^2")
    print(f"z_acc_eq(t) = {zc[0]:.4f} + {zc[1]:.4f}t + {zc[2]:.4f}t^2")

    return x_acc_eq, y_acc_eq, z_acc_eq

""" TODO: 1. get residual errors for speed, acceleration
            2. calculate for t = 6 using acceleration, compare """

def speed_position(x_speed_eq, y_speed_eq, z_speed_eq, time_speed):
    speed_positions = [(x_speed_eq(t_val), y_speed_eq(t_val), z_speed_eq(t_val)) for t_val in time_speed]

    return speed_positions

def acc_position(x_acc_eq, y_acc_eq, z_acc_eq, time_acc):
    acc_positions = [(x_acc_eq(t_val), y_acc_eq(t_val), z_acc_eq(t_val)) for t_val in time_acc]

    return acc_positions

t_values = [0, 1, 2, 3, 4, 5, 6]

x_speed_eq, y_speed_eq, z_speed_eq = speed_slr(x, y, z, t)
x_acc_eq, y_acc_eq, z_acc_eq = acc_pr(x, y, z, t)

speed_pos = speed_position(x_speed_eq, y_speed_eq, z_speed_eq, t_values)
acc_pos = acc_position(x_acc_eq, y_acc_eq, z_acc_eq, t_values)

for i, t_val in enumerate(t_values):
    print(f"At t = {t_val}: x,y,z (speed): {speed_pos[i][0]:.4f}, {speed_pos[i][1]:.4f}, {speed_pos[i][2]:.4f}")
for i, t_val in enumerate(t_values):
    print(f"At t = {t_val}: x,y,z (acc): {acc_pos[i][0]:.4f}, {acc_pos[i][1]:.4f}, {acc_pos[i][2]:.4f}")

print(np.cov(t,x)[0,1]/np.var(t))
print(np.var(t))
print(np.cov(t,x))
print(-2.53/2.91)

print(math.sqrt(1.0276**2+1.5357**2+1.2819**2))
