import numpy as np
import matplotlib.pyplot as plt


def plot_xyz(xyz_list):
    x_list = [x for x, _, _ in xyz_list]
    y_list = [y for _, y, _ in xyz_list]
    z_list = [z for _, _, z in xyz_list]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x_list, y_list, z_list)  # Connect each point

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for i, (x, y, z) in enumerate(xyz_list):
        ax.text(x, y, z, str(i + 1), color="skyblue")

    plt.show()
    fig.savefig("./ass1/out/3d_plot.png")


def main():
    positions = np.array(
        [
            [2.0, 0.0, 1.0],
            [1.08, 1.68, 2.38],
            [-0.83, 1.82, 2.49],
            [-1.97, 0.28, 2.15],
            [-1.31, -1.51, 2.59],
            [0.57, -1.91, 4.32],
        ]
    )
    plot_xyz(positions)


if __name__ == """__main__""":
    main()
