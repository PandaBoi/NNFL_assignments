import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from math import pi
#===================================================

def preprocess(data):

	for i in range(np.shape(data)[1] - 1):

		data[:,i] = (data[:,i] - data[:,i].mean())/data[:,i].std()

	X = data[:,:-2]
	Y = data[:,-1] - 1
	

	return X,Y

def calc_means(x,y):
	means = np.zeros([2,np.shape(x)[1]])

	for i in range(2):
		idx = np.where(y==i)
		for m in range(np.shape(x)[1]):
			means[i,m] = x[idx,m].mean()
			
			

	return means

def calc_std(x,y):

	std = []

	for i in range(2):
		idx = np.where(y==i)
		std.append(np.cov(x[idx].T))

	std = np.array(std)


		

	return std

def test_train_split(X, y,p = 0.6):
	
	indices = np.arange(len(y))
	np.random.shuffle(indices)
	slice_portion = int(p*len(y)-1)
	
	train_slice, test_slice = indices[:slice_portion], indices[slice_portion+1:]
	X_train, X_test, y_train, y_test = X[train_slice], X[test_slice], y[train_slice], y[test_slice]
	# print(X_test,y_test)
	return X_train, X_test, y_train, y_test

def measure_specs( y, y_output):

	TP = 0.0
	FP = 0.0
	TN = 0.0
	FN = 0.0

	


	# print(y)
	for i in range(len(y_output)): 
		if y[i]==y_output[i]==1:
		   TP += 1
		if y_output[i]==1 and y[i]!=y_output[i]:
		   FP += 1
		if y[i]==y_output[i]==0:
		   TN += 1
		if y_output[i]==0 and y[i]!=y_output[i]:
		   FN += 1
	print(TP,FP,FN,TN)
	return (TP,FP,FN,TN)


def LRT(data):

	X,Y = preprocess(data)
	x_train,x_test, y_train,  y_test = test_train_split(X,Y,p = 0.6)
	x_means = calc_means(x_train,y_train)
	x_std = calc_std(x_train,y_train)

	y_pred = []
	n = np.shape(X)[1]
	p_y1 = len(np.where(y_train==0))/len(y_train)
	print(np.where(y_train==0))
	p_y2 = len(np.where(y_train==1))/len(y_train)
	comp_val = p_y2/p_y1
	# print(np.linalg.norm(x_std[1]))
	for i in range(len(x_test)):
		x_t = x_test[i]
		p_x_y1 = (1/(pow(2*pi,n/2)*np.linalg.norm(x_std[0]))) * np.exp( -0.5 * np.dot(np.dot((x_t - x_means[0]),np.linalg.inv(x_std[0])),(x_t - x_means[0])))
		p_x_y2 = (1/(pow(2*pi,n/2)*np.linalg.norm(x_std[1]))) * np.exp( -0.5 * np.dot(np.dot((x_t - x_means[1]),np.linalg.inv(x_std[1])),(x_t - x_means[1])))
		# print(p_x_y1)
		if p_x_y1/p_x_y2 > comp_val:
			y_pred.append(0)
		else:
			y_pred.append(1)

	measure_specs(y_test, y_pred)

data = pd.read_excel('data3.xlsx',header = None)
data = np.array(data)
# print(data)
print(np.shape(data))
LRT(data)