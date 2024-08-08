# logistic regression

import copy
import numpy as np
import matplotlib.pyplot as plot
import matplotlib

font = {'size' : 14}
matplotlib.rc('font', **font)

# Determine the objective -----


# Setup the training data -----

X_train = np.array([[0.25, 1.5],[0.5, 0.5], [2, 0.5],[1.25, 1],[3, 1],[2, 2],[1, 2.5], [3, 2.5]])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

m, n = X_train.shape
plot_split = y_train>0.5

for i in range(m):
	if plot_split[i]:
		plot.scatter(X_train[i][0], X_train[i][1], marker='x', c = 'r', label='positive')
	else:
		plot.scatter(X_train[i][0], X_train[i][1], marker='o', c = 'g', label='negative')
# plot.legend(loc='upper right')
plot.xlabel('feature 1')
plot.ylabel('feature 2')
ax = plot.gca()
ax.set_xlim(0, 3.5)
ax.set_ylim(0, 3)
plot.show()

# Design the model -----

def model(x, w, b):
	z = np.dot(x, w) + b
	return z

def sigmoid(z):
	g = 1/(1+np.exp(-z))
	return g

# Design the cost function -----

def logistic_cost_function(X_train, y_train, model, activation, w, b, lam=1):
	m, n = X_train.shape
	cost = 0.0
	for i in range(m):
		z_i = model(X_train[i],w,b)
		y_pred = activation(z_i)
		cost += -y_train[i]*np.log(y_pred)-(1-y_train[i])*np.log(1-y_pred)
	cost = cost/m
	# return cost

	reg_cost = 0.0
	for j in range(n):
		reg_cost += w[j]**2
	reg_cost = lam/(2*m) * reg_cost

	return cost + reg_cost

w = np.array([1,1])
b = -3

cost = logistic_cost_function(X_train, y_train, model, sigmoid, w, b)
print('cost:', cost)

# Optimise the model -----

def gradient_function(X_train, y_train, model, activation, w, b, lam=1):
	m, n = X_train.shape
	dJ_dw = np.zeros((n,))         # double check dimensions
	dJ_db = 0

	y_pred = np.zeros((m,))

	# loop through samples i, from 0 to m-1
	for i in range(m):
		y_pred[i] = activation(model(X_train[i],w,b))

		# loop through features j, from 0 to n-1
		for j in range(n):
			dJ_dw[j] = dJ_dw[j] + (y_pred[i] - y_train[i])*X_train[i, j]
		dJ_db = dJ_db + (y_pred[i] - y_train[i])
	dJ_dw = dJ_dw / m
	dJ_db = dJ_db / m

	for j in range(n):
		dJ_dw[j] += lam/m*w[j]

	return dJ_dw, dJ_db

w = np.array([1, 1])
b = -3

dJ_dw, dJ_db = gradient_function(X_train, y_train, model, sigmoid, w, b)
print('dJ_dw: ', dJ_dw)
print('dJ_db: ', dJ_db)

def gradient_descent(X_train, y_train, w_init, b_init, alpha, N_iterations, model, activation, cost_function, gradient_function):

    J_log = []
    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(N_iterations):

        dJ_dw, dJ_db = gradient_function(X_train, y_train, model, activation, w, b)

        w = w - alpha * dJ_dw
        b = b - alpha * dJ_db

        if i < 100000:
            J_log.append(cost_function(X_train, y_train, model, activation, w, b))

    return w, b, J_log

n = X_train.shape[1]
print(n)
w_init = np.zeros((n,))
b_init = 0.0
N_iterations = 10000
alpha = 1

w_final, b_final, J_log = gradient_descent(X_train, y_train, w_init, b_init, alpha, N_iterations, model, sigmoid, logistic_cost_function, gradient_function)

f04 = plot.figure(4)
plot.plot(J_log)
plot.xlabel('number of iterations', fontsize=14)
plot.ylabel('cost function', fontsize=14)
plot.show()

print(f'w: {w_final}, b: {b_final}')
print(f'final cost function: {J_log[-1]}')

x_bound = np.arange(0,3.6,0.1)
y_bound = -(w_final[0]/w_final[1])*x_bound-b_final/w_final[1]

plot.plot(x_bound,y_bound, linestyle='dashed')
for i in range(m):
	if plot_split[i]:
		plot.scatter(X_train[i][0], X_train[i][1], marker='x', c = 'r', label='positive')
	else:
		plot.scatter(X_train[i][0], X_train[i][1], marker='o', c = 'g', label='negative')
# plot.legend(loc='upper right')
plot.xlabel('feature 1')
plot.ylabel('feature 2')
ax = plot.gca()
ax.set_xlim(0, 3.5)
ax.set_ylim(0, 3)
plot.show()







