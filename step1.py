import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x = [2, 1.08, -0.83, -1.97, -1.31, 0.57]
y = [0, 1.68, 1.82, 0.28, -1.51, -1.91]
z = [1, 2.38, 2.49, 2.15, 2.59, 4.32 ]

# Set up a 3D plot
# Set up a 3D plot with point labels including coordinates
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data with a label for the legend
ax.plot(x, y, z, marker='o', label='Trajectory')

# Highlight the final point and include a label for the legend

# Add point labels with coordinates
for i, (xi, yi, zi) in enumerate(zip(x, y, z), start=0):
    label = f'{i} ({xi:.2f}, {yi:.2f}, {zi:.2f})'
    ax.text(xi, yi, zi, label, color='blue', ha='right', fontsize=7)

# Set axis limits
ax.set_xlim([-2, max(x)])
ax.set_ylim([-2, max(y)])
ax.set_zlim([-2, max(z)])

# Customize the plot
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Trajectory of the Drone')

# Display the legend
ax.legend()

# Display the plot
plt.show()
