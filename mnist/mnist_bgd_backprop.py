
import numpy as np
import matplotlib.pyplot as plt
import mnist

# Setup the training data -----

x_train = mnist.train_images()
y_train = mnist.train_labels()

m, rows, cols = x_train.shape

print(x_train.shape)	# (60000, 28, 28)
print(y_train.shape)	# (60000,)

print(m, rows, cols)

count = 0
plt.imshow(x_train[count], cmap='gray')
plt.title(y_train[count])
plt.xlabel('columns')
plt.ylabel('rows')
plt.colorbar()
# plt.show()

# print(x_train[0][10])

x_train = x_train/255

# print(x_train[0][10])

# Design the model -----

x_train_flat = x_train.reshape(m, -1)
print(x_train_flat.shape)

input_size = x_train_flat.shape[1]	# 784
hidden_size = 128
output_size = 10

# setup the initial weights and biases

W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randint(low = 0, high = 10, size = hidden_size)

W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randint(low = 0, high = 10, size = output_size)

print(W1.shape)
print(b1.shape)
print(W2.shape)
print(b2.shape)

# print(W1)

def relu(x):
	return np.maximum(0,x)

def softmax(x):
	exponents = np.exp(x - np.max(x, axis=1, keepdims=True))
	return exponents / np.sum(exponents, axis=1, keepdims=True)

alpha = 0.2
epochs = 200

for epoch in range(epochs):

	# Forward propagation
	z1 = np.dot(x_train_flat, W1) + b1
	a1 = relu(z1)
	z2 = np.dot(a1, W2) + b2
	y_pred = softmax(z2)

	# Compute loss
	loss = -np.log(y_pred[range(x_train_flat.shape[0]),y_train]).mean()
	print(f'epoch: {epoch}, loss: {loss}')

	# Back propagation
	dy_pred = y_pred
	dy_pred[range(x_train_flat.shape[0]), y_train] -= 1
	dy_pred /= x_train_flat.shape[0]

	dW2 = np.dot(a1.T, dy_pred)
	db2 = np.sum(dy_pred, axis=0)
	da1 = np.dot(dy_pred, W2.T)
	dz1 = da1*(z1>0)
	dW1 = np.dot(x_train_flat.T, dz1)
	db1 = np.sum(dz1, axis = 0)

	W1 = W1 - alpha * dW1
	b1 = b1 - alpha * db1
	W2 = W2 - alpha * dW2
	b2 = b2 - alpha * db2

np.save('hwd_W1.npy', W1)
np.save('hwd_b1.npy', b1)
np.save('hwd_W2.npy', W2)
np.save('hwd_b2.npy', b2)


















