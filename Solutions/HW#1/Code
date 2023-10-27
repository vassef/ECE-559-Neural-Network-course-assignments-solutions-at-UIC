import numpy as np
import matplotlib.pyplot as plt


# Define the neural network architecture and weights
input_weights = np.array([[1, -1, -1], [-1, -1, 0]])
hidden_bias = np.array([1, 1, 0])
output_weights = np.array([1, 1, -1])
output_bias = -1.5

# Define the step activation function
def step_activation(x):
    return np.where(x >= 0, 1, 0)

# Number of data points
num_points = 1000

# Generate random points in the range [-2, 2] for both dimensions
random_points = np.random.uniform(low=-2, high=2, size=(num_points, 2))

# Calculate the network's output for each random point
hidden_layer_output = step_activation(np.dot(random_points, input_weights) + hidden_bias)
network_output = step_activation(np.dot(hidden_layer_output, output_weights) + output_bias)

# Separate points based on network output
blue_points = random_points[network_output == 0]
red_points = random_points[network_output == 1]

# Plot blue and red points
plt.scatter(blue_points[:, 0], blue_points[:, 1], c='blue', s=5, label='Output 0')
plt.scatter(red_points[:, 0], red_points[:, 1], c='red', s=5, label='Output 1')

# Plot decision boundary (estimated)
xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_hidden_layer_output = step_activation(np.dot(grid_points, input_weights) + hidden_bias)
grid_network_output = step_activation(np.dot(grid_hidden_layer_output, output_weights) + output_bias)
decision_boundary_red = grid_points[grid_network_output == 1]
decision_boundary_blue = grid_points[grid_network_output == 0]
plt.scatter(decision_boundary_blue[:, 0], decision_boundary_blue[:, 1], c='gray', s=0.5, alpha=0.5, label='DB for output 1')
plt.scatter(decision_boundary_red[:, 0], decision_boundary_red[:, 1], c='green', s=0.5, alpha=0.5, label='DB for output 0')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Neural Network Output and Decision Boundary')
plt.legend(loc = 4)
plt.grid(True)
plt.show()

