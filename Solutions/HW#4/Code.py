# -*- coding: utf-8 -*-
"""04-674579894-Vassef.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1esZ_AgUhryoPVm_83Dn53zKcTgoStIT1
"""

import numpy as np
import matplotlib.pyplot as plt

# Part 1: Generate n random numbers uniformly in [0, 1]
n = 300
x = np.random.rand(n)

# Part 2: Generate n random numbers uniformly in [-1/10, 1/10]
nu = np.random.uniform(-1/10, 1/10, n)

# Part 3: Calculate di = sin(20xi) + 3xi + νi
d = np.sin(20 * x) + 3 * x + nu

# Part 3: Plot (xi, di)
plt.scatter(x, d, label='Data Points', color='blue')
plt.xlabel('x')
plt.ylabel('d')
plt.title('Data Points (xi, di)')
plt.grid(True)
plt.legend()
plt.show()

# Define neural network parameters
N = 24  # Number of hidden neurons
input_size = 1
output_size = 1

def BP(lr, ct = 1e-6):
    # Initialize weights and biases
    w = np.random.rand(3 * N + 1)

    w_input = w[:2*N]
    w_output = w[2*N:]

    # Learning rate and convergence threshold
    eta = lr
    convergence_threshold = ct

    # Initialize variables for backpropagation
    epoch = 0
    mse_values = []

    # Backpropagation Algorithm
    while True:
        mse = 0
        for i in range(n):
            # Forward pass
            z = np.dot(w_input[:N], x[i]) + w_input[N:2*N]
            hidden_input = np.tanh(z)    # Using tanh activation for hidden neurons
            output = np.dot(w_output, np.hstack((hidden_input, 1)))
            # Calculate error
            error = d[i] - output
            mse += error ** 2

            # Backward pass
            m = (1 - np.tanh(z) ** 2)
            mm = np.array([m[i]*w_output[i] for i in range(m.shape[0])])
            delta_input = error * mm

            # Update weights and biases
            w_output += eta * error * np.hstack((hidden_input, 1)) # correct
            w_input[:N] += eta * delta_input * x[i] # Weights
            w_input[N:2*N] += eta * delta_input * 1 # Biases

        mse /= n
        mse_values.append(mse)
        epoch += 1

        if epoch % 50 == 0:
            print(f'MSE, and lr for epoch {epoch} is {mse_values[-1]}, and {eta}')

        # Check for convergence
        if epoch > 1 and abs(mse_values[-1] - mse_values[-2]) < convergence_threshold and mse < 0.01:
            break
        # elif epoch > 1 and abs(mse_values[-1] - mse_values[-2]) < convergence_threshold and mse > 0.1:
        #     eta *= 1.1

        # Adjust learning rate if needed
        if epoch > 1 and mse > mse_values[-2]:
            eta *= 0.9  # Reduce learning rate by 10% when MSE increases

        # Optional: Limit the minimum learning rate to prevent it from becoming too small
        min_eta = 1e-6
        if eta < min_eta:
            eta = min_eta

    # Part 4: Plot number of epochs vs. MSE
    plt.plot(range(1, epoch + 1), mse_values, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Number of Epochs vs. MSE (Backpropagation)')
    plt.grid(True)
    plt.show()

    # Part 5: Plot the fitted curve f(x; w0)
    x_range = np.linspace(0, 1, 1000)
    fitted_curve = []

    for x_val in x_range:
        z = np.dot(w_input[:N], x_val) + w_input[N:2*N]
        hidden_input = np.tanh(z)    # Using tanh activation for hidden neurons
        #print(hidden_input)
        output = np.dot(w_output, np.hstack((hidden_input, 1)))#[0]  # Output layer with reshaping
        fitted_curve.append(output)

    # Plot the curve f(x; w0) on top of the data points
    plt.scatter(x, d, label='Data Points', color='blue')
    plt.plot(x_range, fitted_curve, label='Fitted Curve', color='red')
    plt.xlabel('x')
    plt.ylabel('d')
    plt.title('Fitted Curve vs. Data Points')
    plt.grid(True)
    plt.legend()
    plt.show()


print(f'******************** BP reults for lr = 0.005 ************************')
BP(0.005)
print(f'******************** BP reults for lr = 0.01 ************************')
BP(0.01)

# Part 6: Pseudocode for training algorithm
"""
Initialize weights and biases w
Initialize learning rate eta
Initialize convergence threshold epsilon
Initialize epoch = 0
Initialize list mse_values
Repeat
    Initialize mse = 0
    For each data point (xi, di) in the dataset
        Compute forward pass:
            Calculate hidden_input using tanh activation
            Calculate output using linear activation
        Calculate error = di - output
        Update mse += error^2
        Compute backward pass:
            Calculate delta_input for the weights associated with the input x[i] and bias, 1, to the hidden layer.
            Calculate delta_output for the weights associated with the hidden layer to the output layer.
        Update weights and biases:
            Update input layer weights and bias
            Update output layer weights and bias
    Calculate mse /= n
    Append mse to mse_values
    Increment epoch by 1
    Check for convergence: If epoch > 1 and |mse_values[-1] - mse_values[-2]| < epsilon, break
Until convergence
"""

