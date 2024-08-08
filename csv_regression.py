
import matplotlib.pyplot as plot

import csv

csv_file = open('dose_figures.csv')
csv_reader = csv.reader(csv_file)

x_train = [0.0]*20
y_train = [0.0]*20

for i, row in enumerate(csv_reader):
	x_train[i] = float(row[0])
	y_train[i] = float(row[1])

print(x_train)
print(y_train)

f01 = plot.figure(1)
plot.scatter(x_train, y_train, c = 'b')
plot.xlabel('mass of drugs injected i.v. (mg)')
plot.ylabel('mass of drugs in brain (\u03BCg)')
ax = plot.gca()
ax.set_xlim(0, 10)
ax.set_ylim(0, 2500)
# plot.show()

def linear_equation(x, w, b):
	y = w * x + b
	return y

def cost_function(x_train, y_train, model, w, b):
	N_train = len(x_train)
	sum_cost = 0
	for i in range(N_train):
		y_pred = model(x_train[i], w, b)
		cost = (y_pred - y_train[i])**2
		sum_cost = sum_cost + cost
	sum_cost = (1/(2*N_train)) * sum_cost

	return sum_cost

def gradient(x_train, y_train, model, w, b):

	N_train = len(x_train)
	dJ_dw = 0
	dJ_db = 0

	for i in range(N_train):
		y_pred = model(x_train[i], w, b)
		dJ_dw_i = (y_pred - y_train[i])*x_train[i]
		dJ_db_i = (y_pred - y_train[i])
		dJ_dw += dJ_dw_i
		dJ_db += dJ_db_i
	dJ_dw = dJ_dw / N_train
	dJ_db = dJ_db / N_train

	return dJ_dw, dJ_db

def gradient_descent(x_train, y_train, w_init, b_init, alpha, N_iterations, model, cost_function, gradient_function):

	J_log = []
	p_log = []

	w = w_init
	b = b_init

	for i in range(N_iterations):
		dJ_dw, dJ_db = gradient_function(x_train, y_train, model, w, b)
		w = w - alpha * dJ_dw
		b = b - alpha * dJ_db

		J_log.append(cost_function(x_train, y_train, model, w, b))
		p_log.append([w,b])

	return w, b, J_log, p_log

w_init = 0
b_init = 0

N_iterations = 10
alpha = 0.01

w_final, b_final, J_log, p_log = gradient_descent(x_train, y_train, w_init, b_init, alpha, N_iterations, linear_equation, cost_function, gradient)

N_train = len(x_train)
y_pred = N_train*[0.0]
for i in range(N_train):
	y_pred[i] = linear_equation(x_train[i], w_final, b_final)

print(f'w_final and b_final: {w_final}, {b_final}')


w_log, b_log = list(zip(*p_log))

print(len(J_log))
print(J_log)

f02 = plot.figure(2)
plot.plot(x_train, y_pred, c = 'r', label='model prediction')
plot.scatter(x_train, y_train, c = 'b', label='training data')
plot.xlabel('mass of drugs injected i.v. (mg)')
plot.ylabel('mass of drugs in brain (\u03BCg)')
ax = plot.gca()
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 2500)

f03 = plot.figure(3)
plot.plot(w_log)
plot.xlabel('number of iterations')
plot.ylabel('w parameter')

f04 = plot.figure(4)
plot.plot(b_log)
plot.xlabel('number of iterations')
plot.ylabel('b parameter')

f05 = plot.figure(5)
plot.plot(J_log)
plot.xlabel('number of iterations')
plot.ylabel('cost function')
plot.show()




