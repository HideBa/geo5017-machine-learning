from matplotlib import pyplot as plt


def plot_xyz(xyz_list, title):
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
    fig.savefig(f"./ass1/out/3d_plot_{title}.png")


def plot_single(x, y, title, xlabel, ylabel):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.savefig(f"./ass1/out/{title}.png")
