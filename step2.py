import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as  np

def constant_speed():

    time = np.array([0,1,2,3,4,5])

    xyz = np.array([[2,0,1],
                     [1.08, 1.68, 2.38],
                       [-0.83,1.82,2.49],
                       [-1.97,0.28,2.15],
                       [ -1.31,-1.51,2.59],
                       [ 0.57,-1.91, 4.32]])
    

    x_positions= xyz[:, 0]
    y_positions = xyz[:, 1]
    z_positions = xyz[:, 2]

    # Initialize parameters for linear regression in 3D: velocity and initial position for each dimension
    v_x, c_x = 0, 0  # Velocity and initial position in the x dimension
    v_y, c_y = 0, 0  # Velocity and initial position in the y dimension
    v_z, c_z = 0, 0  # Velocity and initial position in the z dimension
    
    # Learning rate and number of iterations
    alpha = 0.01
    iterations = 1000

    # Gradient Descent for each dimension
    for _ in range(iterations):
        # Predict positions using the current model for each dimension
        predicted_x = v_x * time + c_x
        predicted_y = v_y * time + c_y
        predicted_z = v_z * time + c_z
        
        # Calculate the gradient for each parameter in each dimension
        #partial derivative of the cost function with respect to velocity in x
        grad_v_x = (2 / len(time)) * np.sum((predicted_x - x_positions) * time)
        grad_c_x = (2 / len(time)) * np.sum(predicted_x - x_positions)
        grad_v_y = (2 / len(time)) * np.sum((predicted_y - y_positions) * time)
        grad_c_y = (2 / len(time)) * np.sum(predicted_y - y_positions)
        grad_v_z = (2 / len(time)) * np.sum((predicted_z - z_positions) * time)
        grad_c_z = (2 / len(time)) * np.sum(predicted_z - z_positions)

        
        # Update the parameters for each dimension
        v_x -= alpha * grad_v_x
        c_x -= alpha * grad_c_x
        v_y -= alpha * grad_v_y
        c_y -= alpha * grad_c_y
        v_z -= alpha * grad_v_z
        c_z -= alpha * grad_c_z

    # After the loop, v_x, c_x, v_y, c_y, v_z, c_z will be your estimated parameters
    # Calculate the final predicted positions in each dimension
    final_predictions_x = v_x * time + c_x
    final_predictions_y = v_y * time + c_y
    final_predictions_z = v_z * time + c_z

    # Calculate the residual sum of squares (RSS) error for each dimension
    rss_error_x = np.sum((final_predictions_x - x_positions) ** 2)
    rss_error_y = np.sum((final_predictions_y - y_positions) ** 2)
    rss_error_z = np.sum((final_predictions_z - z_positions) ** 2)
    # Total RSS error is the sum of individual RSS errors
    total_rss_error = rss_error_x + rss_error_y + rss_error_z
# Calculate the next time point (one second after the last time point in the time array)
    next_time_point = time[-1] + 1

    # Calculate the next predicted positions in each dimension
    next_predicted_x = v_x * next_time_point + c_x
    next_predicted_y = v_y * next_time_point + c_y
    next_predicted_z = v_z * next_time_point + c_z

    # Print the next predicted position
    print(f"Next predicted position at time {next_time_point}: (X, Y, Z) = ({next_predicted_x}, {next_predicted_y}, {next_predicted_z})")
    
    return total_rss_error, (v_x, c_x), (v_y, c_y), (v_z, c_z)
    


# Run the function and print the total RSS error and the estimated parameters
total_rss_error, speed_x, speed_y, speed_z = constant_speed()
total_rss_error, speed_x, speed_y, speed_z
print(total_rss_error)

# Extracting the velocity components from the parameters
v_x = speed_x[0]
v_y = speed_y[0]
v_z = speed_z[0]


# Calculating the speed as the magnitude of the velocity vector
speed = np.sqrt(v_x**2 + v_y**2 + v_z**2)
speed
print(speed)