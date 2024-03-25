import math
from os import listdir
import os
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
from scipy.spatial import ConvexHull


class urban_object:
    """
    Define an urban object
    """

    def __init__(self, filenm):
        """
        Initialize the object
        """
        # obtain the cloud name
        self.cloud_name = filenm.split("/\\")[-1][-7:-4]

        # obtain the cloud ID
        self.cloud_ID = int(self.cloud_name)

        # obtain the label
        self.label = math.floor(1.0 * self.cloud_ID / 100)

        # obtain the points
        self.points = read_xyz(filenm)

        # initialize the feature vector
        self.feature = []

    def compute_features(self):
        """
        Compute the features, here we provide two example features. You're encouraged to design your own features
        """
        # calculate the height
        height = np.amax(self.points[:, 2])
        self.feature.append(height)

        # get the root point and top point
        root = self.points[[np.argmin(self.points[:, 2])]]
        top = self.points[[np.argmax(self.points[:, 2])]]

        # construct the 2D and 3D kd tree
        kd_tree_2d = KDTree(self.points[:, :2], leaf_size=5)
        kd_tree_3d = KDTree(self.points, leaf_size=5)

        # compute the root point planar density
        radius_root = 0.2
        count = kd_tree_2d.query_radius(
            root[:, :2], r=radius_root, count_only=True
        )
        root_density = 1.0 * count[0] / len(self.points)
        self.feature.append(root_density)

        # compute the 2D footprint and calculate its area
        hull_2d = ConvexHull(self.points[:, :2])
        hull_area = hull_2d.volume

        # get the hull shape index
        hull_perimeter = hull_2d.area
        shape_index = 1.0 * hull_area / hull_perimeter
        self.feature.append(shape_index)

        # obtain the point cluster near the top area
        k_top = max(int(len(self.points) * 0.005), 100)
        idx = kd_tree_3d.query(top, k=k_top, return_distance=False)
        idx = np.squeeze(idx, axis=0)
        neighbours = self.points[idx, :]

        # obtain the covariance matrix of the top points
        cov = np.cov(neighbours.T)
        w, _ = np.linalg.eig(cov)
        w.sort()

        # calculate the linearity and sphericity
        linearity = (w[2] - w[1]) / (w[2] + 1e-5)
        sphericity = w[0] / (w[2] + 1e-5)
        self.feature += [linearity, sphericity]

        self.feature.append(hull_area)
        # calculate the number of points
        # num_points = len(self.points)
        # self.feature.append(num_points)


def read_xyz(filenm):
    """
    Reading points
        filenm: the file name
    """
    points = []
    with open(filenm, "r") as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points


def feature_preparation(data_path):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    # check if the current data file exist
    data_file = "data.txt"
    if exists(data_file):
        return

    # obtain the files in the folder
    files = sorted(listdir(data_path))

    # initialize the data
    input_data = []

    # retrieve each data object and obtain the feature vector
    for file_i in tqdm(files, total=len(files)):
        # obtain the file name
        file_name = join(data_path, file_i)

        # read data
        i_object = urban_object(filenm=file_name)

        # calculate features
        i_object.compute_features()

        # add the data to the list
        i_data = [i_object.cloud_ID, i_object.label] + i_object.feature
        input_data += [i_data]

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    # write the output to a local file
    data_header = (
        "ID,label,height,root_density,area,shape_index,linearity,sphericity"
    )
    np.savetxt(
        data_file,
        outputs,
        fmt="%10.5f",
        delimiter=",",
        newline="\n",
        header=data_header,
    )


def data_loading(data_file="data.txt"):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(
        data_file, dtype=np.float32, delimiter=",", comments="#"
    )

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)

    return ID, X, y


def data_loading2(data_file="data.txt"):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(
        data_file, dtype=np.float32, delimiter=",", comments="#"
    )

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:6].astype(np.float32)

    return ID, X, y


def feature_visualization(X, fig_path="./ass2/out/fig"):
    """
    Visualize the features
        X: input features. This assumes classes are stored in a sequential manner
    """
    features = [
        "height",
        "root_density",
        "area",
        "shape_index",
        "linearity",
        "sphericity",
    ]
    for i in range(0, 5, 2):
        # initialize a plot
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.title(
            "feature subset visualization of 5 classes", fontsize="small"
        )

        # define the labels and corresponding colors
        colors = [
            "firebrick",
            "grey",
            "darkorange",
            "dodgerblue",
            "olivedrab",
        ]
        labels = ["building", "car", "fence", "pole", "tree"]

        # plot the data with first two features
        for j in range(5):
            ax.scatter(
                X[100 * j : 100 * (j + 1), i],
                X[100 * j : 100 * (j + 1), i + 1],
                marker="o",
                c=colors[j],
                edgecolor="k",
                label=labels[j],
            )

        ax.set_xlabel(f"x1:{features[i]}")
        ax.set_ylabel(f"x2:area{features[i+1]}")
        ax.legend()
        plt.show()
        path = os.path.join(fig_path, f"feat_{i}_feat{i+1}.png")
        fig.savefig(path)
