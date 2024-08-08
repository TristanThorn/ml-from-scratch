# regression test

import matplotlib.pyplot as plot

# determine the objective -----

# Setup training data -----

x_train = [1.0, 2.0, 3.0]	# mg of drugs injected in blood
y_train = [40.0, 60.0, 80.0]	# µg of drugs reaching brain tissue

plot.scatter(x_train, y_train)
plot.xlabel('Mass of drugs injected i.v. (mg)')
plot.ylabel('Mass of drugs in brain (\u03BCg)')
ax = plot.gca()
ax.set_xlim(0,5)
ax.set_ylim(0,100)
# plot.show()

# Design the model -----

# Let's try the model y = w * x + b
# where x is in the input and y is the output

def linear_equation(x, w, b):
	y = w * x + b
	return y

w = 2
b = 5
print(f'(w, b) = ({w},{b})')

N_train = len(x_train)
y_pred = N_train*[0]
for i in range(N_train):
	y_pred[i] = linear_equation(x_train[i], w, b)

# Design the cost function -----

plot.plot(x_train, y_pred, c = 'r', label='model prediction')
plot.scatter(x_train, y_train, c = 'b', label = 'true values')
plot.xlabel('Mass of drugs injected i.v. (mg)')
plot.ylabel('Mass of drugs in brain (\u03BCg)')
ax = plot.gca()
ax.set_xlim(0,5)
ax.set_ylim(0,100)
# plot.show()

def cost_function(x_train, y_train, model, w, b):
	N_train = len(x_train)
	sum_cost = 0
	for i in range(N_train):
		y_pred = model(x_train[i], w, b)
		cost = (y_pred - y_train[i]) ** 2
		sum_cost = sum_cost + cost
	sum_cost = (1/(2*N_train)) * sum_cost
	
	return sum_cost

w = 2
b = 5
cost = cost_function(x_train, y_train, linear_equation, w, b)
print(f'cost for (w, b) = ({w},{b}) is {cost}')

b = 10
cost = cost_function(x_train, y_train, linear_equation, w, b)
print(f'cost for (w, b) = ({w},{b}) is {cost}')

# Optimise the model ----

def compute_gradient(x_train, y_train, model, w, b):
	N_train = len(x_train)
	dJ_dw = 0
	dJ_db = 0

	for i in range(N_train):
		y_pred = model(x_train[i], w, b)
		dJ_dw_i	= (y_pred - y_train[i]) * x_train[i]
		dJ_db_i = (y_pred - y_train[i])
		dJ_dw += dJ_dw_i
		dJ_db += dJ_db_i

	dJ_dw = dJ_dw / N_train
	dJ_db = dJ_db / N_train

	return dJ_dw, dJ_db

w = 2
b = 5
dJ_dw, dJ_db = compute_gradient(x_train, y_train, linear_equation, w, b)
print(f'gradient at (w, b) = ({w},{b}) is {dJ_dw, dJ_db}')

w = 4
b = 7
dJ_dw, dJ_db = compute_gradient(x_train, y_train, linear_equation, w, b)
print(f'gradient at (w, b) = ({w},{b}) is {dJ_dw, dJ_db}')


w = 8
b = 14
dJ_dw, dJ_db = compute_gradient(x_train, y_train, linear_equation, w, b)
print(f'gradient at (w, b) = ({w},{b}) is {dJ_dw, dJ_db}')
cost = cost_function(x_train, y_train, linear_equation, w, b)
print(f'cost for (w, b) = ({w},{b}) is {cost}')

w = 10
b = 20
dJ_dw, dJ_db = compute_gradient(x_train, y_train, linear_equation, w, b)
print(f'gradient at (w, b) = ({w},{b}) is {dJ_dw, dJ_db}')
cost = cost_function(x_train, y_train, linear_equation, w, b)
print(f'cost for (w, b) = ({w},{b}) is {cost}')


w = 20
b = 20
dJ_dw, dJ_db = compute_gradient(x_train, y_train, linear_equation, w, b)
print(f'gradient at (w, b) = ({w},{b}) is {dJ_dw, dJ_db}')
cost = cost_function(x_train, y_train, linear_equation, w, b)
print(f'cost for (w, b) = ({w},{b}) is {cost}')

# Use the model to make predictions -----

w = 20
b = 20
print(f'(w, b) = ({w},{b})')

N_train = len(x_train)
y_pred = N_train*[0]
for i in range(N_train):
	y_pred[i] = linear_equation(x_train[i], w, b)

plot.plot(x_train, y_pred, c = 'r', label='model prediction')
plot.scatter(x_train, y_train, c = 'b', label = 'true values')
plot.xlabel('Mass of drugs injected i.v. (mg)')
plot.ylabel('Mass of drugs in brain (\u03BCg)')
ax = plot.gca()
ax.set_xlim(0,5)
ax.set_ylim(0,100)
plot.show()
