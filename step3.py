import numpy as np
import math

def constant_acceleration():
    time = np.array([0, 1, 2, 3, 4, 5])
    xyz = np.array([
        [2, 0, 1],
        [1.08, 1.68, 2.38],
        [-0.83, 1.82, 2.49],
        [-1.97, 0.28, 2.15],
        [-1.31, -1.51, 2.59],
        [0.57, -1.91, 4.32]
    ])
    
    x_positions = xyz[:, 0]
    y_positions = xyz[:, 1]
    z_positions = xyz[:, 2]

    # Initialize parameters for acceleration, velocity, and position in each dimension
    a_x, v_x, c_x = 0, 0, 0  # Acceleration, initial velocity, and position in the x dimension
    a_y, v_y, c_y = 0, 0, 0  # Acceleration, initial velocity, and position in the y dimension
    a_z, v_z, c_z = 0, 0, 0  # Acceleration, initial velocity, and position in the z dimension
    
    alpha = 0.01  # Learning rate
    iterations = 1000  # Number of iterations

    for _ in range(iterations):
        # Predict positions using the current model for each dimension
        predicted_x = 0.5 * a_x * time**2 + v_x * time + c_x
        predicted_y = 0.5 * a_y * time**2 + v_y * time + c_y
        predicted_z = 0.5 * a_z * time**2 + v_z * time + c_z
        
        # Compute gradients for acceleration, velocity, and position in each dimension
        # Example for x dimension (you'll need to compute these for y and z dimensions as well)
        grad_a_x = (2 / len(time)) * np.sum((predicted_x - x_positions) * time**2)
        grad_v_x = (2 / len(time)) * np.sum((predicted_x - x_positions) * time)
        grad_c_x = (2 / len(time)) * np.sum(predicted_x - x_positions)
        grad_a_y = (2 / len(time)) * np.sum((predicted_y - y_positions) * time**2)
        grad_v_y = (2 / len(time)) * np.sum((predicted_y - y_positions) * time)
        grad_c_y = (2 / len(time)) * np.sum(predicted_y - y_positions)
        grad_a_z = (2 / len(time)) * np.sum((predicted_z - z_positions) * time**2)
        grad_v_z = (2 / len(time)) * np.sum((predicted_z - z_positions) * time)
        grad_c_z = (2 / len(time)) * np.sum(predicted_z - z_positions)
        
        # Update the parameters for each dimension
        a_x -= alpha * grad_a_x
        v_x -= alpha * grad_v_x
        c_x -= alpha * grad_c_x
        a_y -= alpha * grad_a_y
        v_y -= alpha * grad_v_y
        c_y -= alpha * grad_c_y
        a_z -= alpha * grad_a_z
        v_z -= alpha * grad_v_z
        c_z -= alpha * grad_c_z
        # Repeat for y and z dimensions

    # Compute residual errors for each dimension
    residual_error_x = np.sum((0.5 * a_x * time**2 + v_x * time + c_x - x_positions)**2)
    residual_error_y = np.sum((0.5 * a_y * time**2 + v_y * time + c_y - y_positions)**2)
    residual_error_z = np.sum((0.5 * a_z * time**2 + v_z * time + c_z - z_positions)**2)

    print(residual_error_x)
    print(residual_error_y)
    print(residual_error_z)

# Total residual error
    total_residual_error = residual_error_x + residual_error_y + residual_error_z

    # Calculate the next predicted positions in each dimension based on acceleration
    next_time_point = time[-1] + 1
    next_predicted_x = 0.5 * a_x * next_time_point**2 + v_x * next_time_point + c_x
    next_predicted_y = 0.5 * a_y * next_time_point**2 + v_y * next_time_point + c_y
    next_predicted_z = 0.5 * a_z * next_time_point**2 + v_z * next_time_point + c_z

    # Print the next predicted position
    print(f"Next predicted position at time {next_time_point}: (X, Y, Z) = ({next_predicted_x}, {next_predicted_y}, {next_predicted_z})")
    print("total residual error = " + str(total_residual_error))
    
constant_acceleration()