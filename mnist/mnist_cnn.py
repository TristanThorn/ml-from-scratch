from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
from torch import nn

import matplotlib.pyplot as plt


# Determine the objective -----


# Setup the training data -----

train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test_data  = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

img, y_train = train_data[0]
img = img.squeeze()

class_names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

count = 0
plt.imshow(img, cmap='gray')
plt.title(class_names[y_train])
plt.xlabel('columns')
plt.ylabel('rows')
plt.colorbar()
# plt.show()

# Convert your data into a DataLoader object

batch_size = 64

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

print(len(train_dataloader))
print(len(test_dataloader))

# Design the model -----

device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

class NeuralNetwork(nn.Module):
	# def __init__(self):
	# 	super().__init__()
	# 	self.flatten = nn.Flatten()
	# 	self.linear_relu_stack = nn.Sequential(
	# 		nn.Linear(28*28, 128),
	# 		nn.ReLU(),
	# 		nn.Linear(128,10)
	# 		)

	# def forward(self, x):
	# 	# print('forward() called')
	# 	x = self.flatten(x)
	# 	logits = self.linear_relu_stack(x)
	# 	return logits

	def __init__(self):
		super(NeuralNetwork,self).__init__()
		
		# convolution 1
		self.c1 = nn.Conv2d(1, 16, kernel_size=(5,5))
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))

		# convolution 2
		self.c2 = nn.Conv2d(16, 32, kernel_size=(3,3))
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))

		# linear portion
		self.fc1 = nn.Linear(32*5*5, 256)
		self.fc2 = nn.Linear(256,10)

	def forward(self, x):

		out = self.c1(x)			# [batch_size, 16, 24, 24]
		out = self.relu1(out)
		out = self.maxpool1(out)	# [batch_size, 16, 12, 12]

		out = self.c2(out)			# [batch_size, 32, 10, 10]
		out = self.relu2(out)
		out = self.maxpool2(out)		# [batch_size, 32, 5, 5]

		out = out.view(out.size(0),-1)	# [batch_size, 32*5*5=800]
		out = self.fc1(out)			# [batch_size, 256]
		out = self.fc2(out)			# [batch_size, 10]

		return out

model = NeuralNetwork().to(device)


# Design the cost function -----

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# Train the model -----

def train(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		# print('batch:', batch)
		X, y = X.to(device), y.to(device)

		pred = model(X)
		loss = loss_fn(pred, y)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

def test(dataloader, model, loss_fn):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0,0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1)==y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f'Accuracy: {100*correct:>0.1f}%')


epochs = 10
for t in range(epochs):
	print('epoch ', t)
	train(train_dataloader, model, loss_fn, optimizer)
	test(test_dataloader, model, loss_fn)


torch.save(model.state_dict(), 'model.pth')
print('Save pytorch model state to model.pth')


# # Make/test predictions ------

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model.pth'))

print('Load pytorch model states from model.pth')

model.eval()

test(test_dataloader, model, loss_fn)



