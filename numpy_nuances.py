
import numpy as np


a = np.array([1, 2])	# vector
b = np.array([[1, 2]])	# 1 x 2
c = np.array([[1],[2]])	# 2 x 1

print(f'a with shape {a.shape} type {a.dtype}:\n{a}')
print(f'b with shape {b.shape} type {b.dtype}:\n{b}')
print(f'c with shape {c.shape} type {c.dtype}:\n{c}')

d = np.array([[1, 2, 3],[4, 5, 6]])		# 2 x 3 matrix
e = np.array([[1,2],[3,4],[5,6]])			# 3 x 2 matrix

print(f'd with shape {d.shape} type {d.dtype}:\n{d}')
print(f'e with shape {e.shape} type {e.dtype}:\n{e}')

f = np.dot(d,e)
print(f'f with shape {f.shape} type {f.dtype}:\n{f}')

# g = np.dot(d,d)

h = np.dot(a,a)
print(f'h with shape {h.shape} type {h.dtype}:\n{h}')


j = np.ones(4)
k = np.ones((4,))
l = np.ones((4,3))

print(f'j with shape {j.shape} type {j.dtype}:\n{j}')
print(f'k with shape {k.shape} type {k.dtype}:\n{k}')
print(f'l with shape {l.shape} type {l.dtype}:\n{l}')




