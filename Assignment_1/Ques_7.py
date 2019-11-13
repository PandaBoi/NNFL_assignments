import numpy as np	
import pandas as pd
from math import log, exp
import matplotlib.pyplot as plt

#======================================================

def sigmoid(temp):

	ans =0.0
	ans = 1 / (1 + np.exp(-temp))

	return ans

def preprocess(data):

	# print(np.shape(data))
	for i in range(np.shape(data)[-1] -1):

		temp_mean = np.mean(data[:,i])
		temp_std = np.std(data[:,i])
		# print(temp_std)
		data[:,i] = (data[:,i] - temp_mean)/temp_std
		# print(data[:,i])

	X = data[:,0:-2]
	# print(np.max(X))
	Y = data[:,-1] - 1
	bias = np.ones([len(Y),1])
	X = np.append(bias,X,axis = 1)
	Y = np.expand_dims(Y, axis = 1)

	return X, Y

def test_train_split(X, y,p = 0.6):
	
	indices = np.arange(len(y))
	np.random.shuffle(indices)
	slice_portion = int(p*len(y)-1) 
	train_slice, test_slice = indices[:slice_portion], indices[slice_portion+1:]
	X_train, X_test, y_train, y_test = X[train_slice], X[test_slice], y[train_slice], y[test_slice]
	# print(X_test,y_test)
	return X_train, X_test, y_train, y_test

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


def measure_specs(X, y, W):

	TP = 0.0
	FP = 0.0
	TN = 0.0
	FN = 0.0

	y_output = sigmoid(np.dot(X,W))

	y_output = 1 * (y_output > 0.5)
	y = y.flatten()
	y_output = y_output.flatten()

	y_vals = np.array(list(set(y)))
	y_vals = y_vals.astype(int)
	y_output = y_output.astype(int)
	y = y.astype(int)
	# print(y,y_output,y_vals)
	print(len(y_output))
	for i in range(len(y_output)): 
		if y[i]==y_vals[0] and y_output[i]==y_vals[0]:
		   TP += 1
		elif  y[i]==y_vals[0] and y_output[i]!=y[i]:
		   FP += 1
		elif y[i]==y_vals[1] and y_output[i]!=y[i]:
		   FN += 1
		elif y_output[i]==y_vals[1] and y[i]==y_vals[1]:
		   TN += 1
	# print(TP,FP,FN,TN)

	return (TP,FP,FN,TN)


def logistic_regr(data = None,dataa = None,alpha =0.05,epsilon = 1e-10,flag=-1):

	if flag == -1:
		X,y = preprocess(data)
		X_train, X_test, y_train, y_test = test_train_split(X, y, p = 0.6)

	elif flag ==0:
		X_train, X_test, y_train, y_test = test_train_split(X, y, p = 0.6)

	elif flag == 1:
		X_train, X_test, y_train, y_test = 	dataa['X_train'],dataa['X_test'],dataa['y_train'],dataa['y_test']	

	W = np.zeros([np.shape(X_train)[1],1])
	# W = np.random.randn(4, 1)
	epochs = 1000
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

	# a =measure_specs(X_test, y_test, W)
	# print(a)
	# SE = a[3]/(a[2]+a[3])
	# SP = a[0]/(a[0]+a[1])
	# acc = (a[0]+a[3])/sum(a)
	# print(SE,SP,acc)
	return W,X_test,y_test



if __name__ == '__main__':

	data = pd.read_excel('data3.xlsx', header = None)
	data = np.array(data)
	logistic_regr(data = data,alpha = 0.01)
	# plt.plot(j)
	# plt.show()







