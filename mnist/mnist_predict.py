
import numpy as np
import matplotlib.pyplot as plt
import mnist

# Setting up the test data -----

x_test = mnist.test_images()
y_test = mnist.test_labels()

print(x_test.shape)
print(y_test.shape)

x_test = x_test/255
x_test_flat = x_test.reshape(x_test.shape[0], -1)

print(x_test_flat.shape)

# Load the model parameters -----

W1 = np.load('hwd_W1.npy')
b1 = np.load('hwd_b1.npy')
W2 = np.load('hwd_W2.npy')
b2 = np.load('hwd_b2.npy')

# Make predictions -----

def relu(x):
	return np.maximum(0,x)

def softmax(x):
	exponents = np.exp(x - np.max(x, axis=1, keepdims=True))
	return exponents / np.sum(exponents, axis=1, keepdims=True)


z1 = np.dot(x_test_flat, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
y_pred = np.argmax(z2, axis = 1)
accuracy = np.mean(y_pred == y_test)

print(f'accuracy: {accuracy}')

f, axarr = plt.subplots(3,3)
count = 9
for i in range(0,3):
	for j in range(0,3):
		print(i,j)
		axarr[i,j].imshow(x_test[count])
		axarr[i,j].title.set_text(f'prediction: {y_pred[count]}')
		count+=1
f.tight_layout()
plt.show()


















