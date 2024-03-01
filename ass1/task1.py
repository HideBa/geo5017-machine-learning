import numpy as np
from plot import plot_single, plot_xyz
from common import POSITIONS


def main():
    positions = np.array(POSITIONS)
    plot_xyz(positions, "plot_trajectory")

    x = [x for x, _, _ in positions]
    y = [y for _, y, _ in positions]
    z = [z for _, _, z in positions]
    time_stemp = [i for i in range(0, len(positions))]
    plot_single(time_stemp, x, "X", "Time", "X")
    plot_single(time_stemp, y, "Y", "Time", "Y")
    plot_single(time_stemp, z, "Z", "Time", "Z")


if __name__ == """__main__""":
    main()
