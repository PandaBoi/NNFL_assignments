import numpy as np	
import pandas as pd
from math import log, exp
import matplotlib.pyplot as plt

#==========================================

def sigmoid(temp):

	ans =0.0
	ans = 1 / (1 + np.exp(-temp))

	return ans

def compute_J (h_x, y):

	J = 0.0
	for i in range(len(y)):
		try:
			J += y[i]*(log(h_x[i])) +(1-y[i]) * (log(1 - h_x[i]))
		except:
			print('logistic error')
			print(y[i],h_x[i],i)
			exit(0)


	J /= -len(y)

	return J

def del_j (h_x, X, y, idx):

	del_j = 0.0
	# print(h_x,y)
	for i in range(len(y)):
		del_j += (h_x[i] - y[i]) * X[i,idx]

	
	del_j /= len(y)

	return del_j

def logistic_regr(X_train,y_train,alpha =0.05,epsilon = 1e-10):

	

	W = np.zeros([np.shape(X_train)[1],1])

	# W = np.random.randn(4, 1)
	epochs = 3000
	ep = 0
	eps = np.Inf
	cost_vals = []

	h_x = sigmoid(np.dot(X_train,W))
	# print(h_x)

	while eps > epsilon and ep < epochs:
		# print('ep: ',ep)
		
		for j in range(len(W)):

			for i in range(len(y_train)):
				W[j] = W[j] - alpha*((h_x[i] - y_train[i])*X_train[i][j])

		h_x = sigmoid(np.dot(X_train,W))
		cost_vals.append(compute_J(h_x,y_train))
		ep +=1

		if(len(cost_vals)>1):
			eps = np.abs(cost_vals[-1] - cost_vals[-2])

	return W