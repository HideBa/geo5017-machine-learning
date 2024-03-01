import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x = [2, 1.08, -0.83, -1.97, -1.31, 0.57, 2.15]
y = [0, 1.68, 1.82, 0.28, -1.51, -1.91, -4.99]
z = [1, 2.38, 2.49, 2.15, 2.59, 4.32, 4.62]

# Set up a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data
ax.plot(x, y, z, marker='o')

# Customize the plot
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Trajectory of the Drone')

# Display the plot
plt.show()