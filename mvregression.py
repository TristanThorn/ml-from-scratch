# multiple variable linear regression

import numpy as np
import matplotlib.pyplot as plot
import copy

# Determine the objective -----

# Setup the training data -----

X_train = np.random.rand(2000).reshape(1000,2)*60
y_train = (X_train[:, 0]**2)+(X_train[:,1]**2)

f01 = plot.figure(1)
ax = f01.add_subplot(111, projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], y_train, marker='.', color='r')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
# plot.show()

m, n = X_train.shape
print(m)
print(n)


# Design the model -----

def mv_equation(X, w, b):
	y = np.dot(X,w) + b
	return y

w = np.array([10, 10])		# vector with 2 elements
b = 100
y_pred = mv_equation(X_train, w, b)
# print(y_pred)

# y_pred = mv_equation(X_train[0], w, b)
# print(y_pred)


# Design the cost function -----

def cost_function(X_train, y_train, model, w, b):
	m, n = X_train.shape
	cost = 0.0
	for i in range(m):
		y_pred = mv_equation(X_train[i], w, b)
		cost += (y_pred - y_train[i])**2
	cost = cost/(2*m)
	return cost

w = np.array([50, 50])		# vector with 2 elements
b = 0
cost = cost_function(X_train, y_train, mv_equation, w, b)
print(cost)


# Optimise the model -----

def gradient_function(X_train, y_train, model, w, b):
	m, n = X_train.shape
	dJ_dw = np.zeros((n,))
	dJ_db = 0

	# loop through samples i, from 0 to m-1
	for i in range(m):
		y_pred[i] = model(X_train[i],w,b)

		# loop through fetures j, from 0 to n-1
		for j in range(n):
			dJ_dw[j] = dJ_dw[j] + (y_pred[i]-y_train[i])*X_train[i,j]
		dJ_db = dJ_db + (y_pred[i] - y_train[i])
	dJ_dw = dJ_dw/m
	dJ_db = dJ_db/m

	return dJ_dw, dJ_db

dJ_dw, dJ_db = gradient_function(X_train, y_train, mv_equation, w, b)

print(dJ_dw)
print(dJ_db)

def gradient_descent(X_train, y_train, w_init, b_init, alpha, N_iterations, model, cost_function, gradient_function):

	J_log = []
	w = copy.deepcopy(w_init)
	b = b_init

	for i in range(N_iterations):
		dJ_dw, dJ_db = gradient_function(X_train, y_train, model, w, b)
		w = w - alpha * dJ_dw
		b = b - alpha * dJ_db

		if i < 100000:
			J_log.append(cost_function(X_train,y_train,model, w, b))

	return w, b, J_log

# Analyse the prediction performance -----

w_init = np.zeros((2,))
b_init = 0.0
N_iterations = 100
alpha = 0.0001

w_final, b_final, J_log = gradient_descent(X_train, y_train, w_init, b_init, alpha, N_iterations, mv_equation, cost_function, gradient_function)

# use w and b to make predictions
print(f'w_final: {w_final}, b_final{b_final}')

f02 = plot.figure(2)
plot.plot(J_log)
plot.xlabel('number of iterations')
plot.ylabel('cost function')
plot.show()

y_pred = mv_equation(X_train, w_final, b_final)

xs = np.tile(np.arange(61), (61, 1))
ys = np.tile(np.arange(61), (61, 1)).T
zs = xs*w_final[0]+ys*w_final[1]+b_final

f03 = plot.figure(3)
ax = f03.add_subplot(111, projection = '3d')
ax.scatter(X_train[:,0], X_train[:,1], y_train, marker='.', color='r')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.plot_surface(xs,ys,zs, alpha = 0.5)
plot.show()


