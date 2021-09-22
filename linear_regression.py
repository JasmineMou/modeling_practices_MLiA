'''
(p.153 ~ 159)
Linear regression to find best line of fit, also called Ordinary Least Squares:
to find weight variable w:
- minimize the error btw predicted and actual y (squared error), 
- taking the derivative of the squared error with respect to w, and set it to 0 to solve for w.
- weights' formula:
w = (x^T * x)^(-1) * X^T * y

remember to: 
- check determinant before taking the inverse operation: https://www.mathsisfun.com/algebra/matrix-determinant.html
- take the transform of y for the calculation with x
'''

import numpy as np

class linear_regression:
	def __init__(self):
		self.x = None
		self.y = None
		self.weights = []

	def fit(self, x, y):
		self.x = np.mat(x)
		self.y = np.mat(y)
		# handle the exception of singular matrix
		xTx = self.x.T * self.x

		## M1. hand calculate with inverse
		# if np.linalg.det(xTx) == 0.0: 
		# 	print("Matrix of x is singular, cannot do inverse")
		# 	return 
		# self.weights = xTx.I * self.x.T * self.y.T

		## M2. use np.linalg.solve()
		self.weights = np.linalg.solve(xTx, self.x.T * self.y.T)

	def get_weights(self):
		return self.weights

def load_data(path):
	with open(path, 'r') as inputs:
		lines = inputs.readlines()
	x = []
	y = []
	for l in lines:
		splits = [float(i) for i in l.strip().split('\t')]
		x.append(splits[:-1])
		y.append(splits[-1])
	return x, y


data_file = "../MLiA_source_codes/Ch08/ex0.txt"
x, y = load_data(data_file)
model = linear_regression()
model.fit(x, y)
weights = model.get_weights()
print(weights) # should be [[3.00774324], [ 1.69532264]]