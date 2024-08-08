
import numpy as np
import matplotlib.pyplot as plot
import copy

# Determine the objective -----


# Setup the training data -----

X_train = np.random.rand(2000).reshape(1000,2)*60
X_train[:, 1] = 100*X_train[:, 1]
y_train = (X_train[:, 0]**2)+(X_train[:, 1]**2)  

f01 = plot.figure(1)
ax = f01.add_subplot(111, projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], y_train, marker='.', color = 'red')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
plot.show()

def z_score_norm(X_raw):
	mu = np.mean(X_raw, axis=0)
	sigma = np.std(X_raw, axis=0)
	X_norm = (X_raw - mu)/sigma
	return X_norm, mu, sigma

print(X_train)
X_norm, mu, sigma = z_score_norm(X_train)
print(X_norm)

f02 = plot.figure(2)
ax = f02.add_subplot(111, projection='3d')
ax.scatter(X_norm[:,0], X_norm[:,1], y_train, marker='.', color = 'red')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
plot.show()


# Design the model -----

def mv_equation(x, w, b):
    y = np.dot(x, w) + b
    return y

# Design the cost function -----

def cost_function(X_train, y_train, model, w, b, lam=1):

    m, n = X_train.shape  # total number of samples
    cost = 0.0
    for i in range(m):
        y_pred = model(X_train[i],w,b)
        cost = cost + (y_pred - y_train[i])**2
    cost = cost / (2 * m)
    # return cost

    reg_cost = 0.0
    for j in range(n):
        reg_cost += w[j] ** 2
    reg_cost = lam / (2 * m) * reg_cost
    return cost + reg_cost
# Optimise the model -----

def gradient_function(X_train, y_train, model, w, b, lam=1):
    m, n = X_train.shape
    dJ_dw = np.zeros((n,))  # double check dimensions
    dJ_db = 0

    y_pred = np.zeros((m,))
    
    # loop through samples i, from 0 to m-1
    for i in range(m):
        y_pred[i] = model(X_train[i],w,b)

        # loop through features j, from 0 to n-1
        for j in range(n):
            dJ_dw[j] = dJ_dw[j] + (y_pred[i] - y_train[i])*X_train[i, j]
        dJ_db = dJ_db + (y_pred[i] - y_train[i])
    dJ_dw = dJ_dw / m
    dJ_db = dJ_db / m

    for j in range(n):
        dJ_dw[j] += lam / m * w[j]

    return dJ_dw, dJ_db


def gradient_descent(X_train, y_train, w_init, b_init, alpha, N_iterations, model, cost_function, gradient_function):

    J_log = []
    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(N_iterations):

        dJ_dw, dJ_db = gradient_function(X_train, y_train, model, w, b)

        w = w - alpha * dJ_dw
        b = b - alpha * dJ_db

        if i < 100000:
            J_log.append(cost_function(X_train, y_train, model, w, b))

    return w, b, J_log

w_init = np.zeros((2,))
b_init = 0.0
# N_iterations = 1000
N_iterations = 100
# alpha = 0.00000001
alpha = 0.1

w_final, b_final, J_log = gradient_descent(X_norm, y_train, w_init, b_init, alpha, N_iterations, mv_equation, cost_function, gradient_function)

# Analyse prediction performance -----

f02 = plot.figure(2)
plot.plot(J_log)
plot.xlabel('number of iterations')
plot.ylabel('cost function')
plot.show()

print(f'w: {w_final}, b: {b_final}')
print(f'final cost function: {J_log[-1]}')
y_pred = mv_equation(X_norm, w_final, b_final)

# xs = np.tile(np.arange(61), (61,1))
# ys = np.tile(np.arange(0,6001,100), (61,1)).T
xs = np.tile(np.arange(-1.5,1.55,0.05), (61,1))
ys = np.tile(np.arange(-1.5,1.55,0.05), (61,1)).T
zs = xs*w_final[0]+ys*w_final[1]+b_final
print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(b_final, w_final[0], w_final[1]))

f03 = plot.figure(3)
ax = f03.add_subplot(111, projection='3d')
ax.scatter(X_norm[:,0], X_norm[:,1], y_train, marker='.', color='red')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.plot_surface(xs,ys,zs, alpha=0.5)
plot.show()
