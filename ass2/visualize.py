import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from preprocess import read_xyz


def main(directory):
    class_labels = ["building", "car", "fence", "pole", "tree"]
    class_colors = ["blue", "red", "black", "gray", "green"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i, filename in enumerate(sorted(os.listdir(directory))):
        print("reading: ", filename)
        if filename.endswith(".xyz"):
            filepath = os.path.join(directory, filename)
            class_index = i // 100
            nth_in_class = i % 100
            if nth_in_class > 30:
                continue
            data = np.loadtxt(filepath, delimiter=" ")

            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]

            ax.scatter(
                x,
                y,
                z,
                c=class_colors[class_index],
                marker="o",
                label=class_labels[class_index],
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Point Cloud Visualization")

    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

    # ax.legend()

    # Display the plot
    plt.show()


def plot(directory):
    class_labels = ["building", "car", "fence", "pole", "tree"]
    class_colors = {
        0: "blue",
        1: "red",
        2: "black",
        3: "gray",
        4: "green",
    }
    fig = go.Figure()

    for file_name in os.listdir(directory)[:500]:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            file_index = int(file_name.split(".")[0])
            class_index = file_index // 100
            print("reading: ", file_name)
            print("class_index: ", class_index)
            nth_in_class = file_index % 100
            # if nth_in_class > 30:
            #     continue
            points = read_xyz(file_path)

            # Add the points to the figure with the associated class color
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2, color=class_colors.get(class_index, "blue")
                    ),
                )
            )  # Default to blue if class not found

    # Update the layout for a better viewing experience
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"
        ),
    )

    # Show the interactive figure
    fig.show()


if __name__ == "__main__":
    # specify the data folder
    """ "Here you need to specify your own path"""
    path = "./ass2/pointclouds-500"
    # main(path)
    plot(path)
