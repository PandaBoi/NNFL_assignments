import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import Ques_7
import logistic_reg as lr
#complete metric calc

total = 0
denom = 0

def sigmoid(temp):

	ans =0.0
	ans = 1 / (1 + np.exp(-temp))

	return ans



def preprocess(data):

	for i in range(np.shape(data)[1] - 1):

		data[:,i] = (data[:,i] - data[:,i].mean())/data[:,i].std()

	X = data[:,:-1]
	Y = data[:,-1] -1
	bias = np.ones([len(Y),1])
	X = np.append(bias,X,axis = 1)
	

	
	return X,Y

def data_segment(X,Y,i,j =-1):

	if j == -1:
		first_class = np.where(Y == float(i))
		second_class = np.where(Y!=float(i))
	else:
		first_class = np.where(Y == float(i))
		second_class = np.where(Y == float(j))
	new_Y = np.zeros(len(Y))

	new_Y[first_class] = 2
	new_Y[second_class] = 1
	# print(new_Y)
	new_data = np.append(X,np.expand_dims(new_Y,axis =1 ),axis = 1)

	if j!= -1:
		new_data = new_data[new_data[:,-1]>0]
	new_data[:,-1] = new_data[:,-1] -1

	return new_data



def test_train_split(X, y,p = 0.6):
	
	indices = np.arange(len(y))
	np.random.shuffle(indices)
	slice_portion = int(p*len(y)-1) 
	train_slice, test_slice = indices[:slice_portion], indices[slice_portion+1:]
	X_train, X_test, y_train, y_test = X[train_slice], X[test_slice], y[train_slice], y[test_slice]
	# print(X_test,y_test)
	return X_train, X_test, y_train, y_test
	

def OnevAll (data):

	X,Y = preprocess(data)

	X_train, X_test, y_train, y_test = test_train_split(X, Y,p =0.6)

	W = []
	
	for i in range(3):
		data_part = data_segment(X_train,y_train,i)

	
		w = lr.logistic_regr(data_part[:,:-1],data_part[:,-1],alpha = 0.009)
		W.append(w)
		
	
	temp =np.zeros([len(y_test),3])
	for l,(x,y) in enumerate(zip(X_test,y_test)):
		
		for xx,w in enumerate(W):
			y_output = sigmoid(np.dot(x,w))
			
			temp[l,xx] = y_output
	results = np.argmax(temp,axis = 1)

	cc = np.zeros([3,3])
	for yy,yyy in zip(results,y_test):
		cc[int(yy)][int(yyy)] +=1
	print(cc)
	print(len(y_test))
	# print(W)

def OnevOne (data):
		
	
	X,Y = preprocess(data)
	X_train, X_test, y_train, y_test = test_train_split(X, Y,p =0.6)
	W = []
	
	
	for i in range(3):

		for j in range(i+1,3):
			print(i,j)
			data_part = data_segment(X_train,y_train,i,j)
			# print(data_part[:,-1])
		#do logistic regr for one by all
			w = lr.logistic_regr(data_part[:,:-1],data_part[:,-1],alpha = 0.05)
			W.append(w)

	temp =np.zeros([len(y_test),3])
	for l,(x,y) in enumerate(zip(X_test,y_test)):
		
		for xx,w in enumerate(W):
			y_output = sigmoid(np.dot(x,w))
			
			temp[l,xx] = (y_output>0.55)
	# print(temp)

	cc = np.zeros([3,3])
	p = 0
	for i in range(len(y_test)):
		# temp[0] temp[1] temp [2]
		if temp[i][0] == 1 and temp[i][1] == 1: p =1
		if temp[i][0] == 0 and temp[i][2] == 1: p =2
		if temp[i][1] == 0 and temp[i][2] == 0: p =3

		cc[p-1][int(y_test[i])]+=1
		
	print(cc)
	print(y_test)
	print(len(y_test))



		



if __name__ == '__main__':
	data = pd.read_excel('data4.xlsx',header = None)
	data = np.array(data)
	OnevOne(data)