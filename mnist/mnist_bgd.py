
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader


import matplotlib.pyplot as plt

# Determine the objective -----


# Setup the training data -----

training_data = datasets.MNIST(
	root='data',
	train=True,
	download=True,
	transform=ToTensor())

test_data = datasets.MNIST(
	root='data',
	train=False,
	download=True,
	transform=ToTensor())

print(training_data)
print(test_data)

# raw_data, target = training_data[0]
# img = raw_data.squeeze()

# count = 0
# plt.imshow(img, cmap='gray')
# plt.xlabel('columns')
# plt.ylabel('rows')
# plt.colorbar()
# plt.show()

class_names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

# f, axarr = plt.subplots(3,3)
# count = 0
# for i in range(0,3):
#     for j in range(0,3):
#         print(i,j)
#         img, y_train = training_data[count]
#         img = img.squeeze()
#         axarr[i,j].imshow(img, cmap='gray')
#         axarr[i,j].title.set_text(class_names[y_train])
#         count += 1
# for ax in axarr.flat:
# 	ax.set(xlabel='columns', ylabel='rows')
# for ax in axarr.flat:
# 	ax.label_outer()
# f.tight_layout()
# plt.show()

batch_size = 64		# one of the parameters that we can adjust

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in train_dataloader:
# 	print(X.shape, y)

# Design the model -----

device = (
	'cuda'
	if torch.cuda.is_available()
	else 'mps'
	if torch.backends.mps.is_available()
	else 'cpu')
print(device)

class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 128),
			nn.ReLU(),
			nn.Linear(128,10)
			)

	def forward(self, x):
		# print('forward() called')
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits

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
print('saved pytorch model state to model.pth')

# Make/test predictions ------

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model.pth'))

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
	x = x.to(device)
	pred = model(x)
	predicted, actual = class_names[pred[0].argmax(0)], class_names[y]
	print(f'Predicted: "{predicted}", actual: "{actual}"')











