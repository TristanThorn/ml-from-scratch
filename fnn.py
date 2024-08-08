import numpy as np

# Define test data X
X = np.array([
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [1, 1, -1, -1, -1],
    [1, 1, 1, 1, 1],
    [2, 2, -2, -2, -2],
    [3, 3, -3, -3, -3],
    [4, 4, -4, -4, -4]
])

# Define weights and biases for hidden layer
W1 = np.array([
    [1, 0, 1, 0],
    [1, 1, 1, 1],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1]
])
b1 = np.array([1, 2, 1, 2])

# Define weights and biases for output layer
W2 = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3]
])
b2 = np.array([1, 2])

# Define ReLU activation function
def relu(Z):
    return np.maximum(0, Z)

# Perform forward propagation
Z1 = np.dot(X, W1) + b1
A1 = relu(Z1)
Z2 = np.dot(A1, W2) + b2

print("Z[2] =")
print(Z2)
