import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Ques_7
# import Ques_8
import logistic_reg as lr
#=====================================
total = 0.0
denom = 0.0
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

	X = data[:,0:-1]
	# print(np.max(X))
	Y = data[:,-1] - 1
	bias = np.ones([len(Y),1])
	X = np.append(bias,X,axis = 1)

	return X, Y



def measure_specs(X, y, W):
	global total,denom
	TP = 0.0
	FP = 0.0
	TN = 0.0
	FN = 0.0
	y_vals = list(set(y))
	# print(len(y))
	y_output = sigmoid(np.dot(X,W))

	y_output = 1 * (y_output > 0.5)
	y_output = y_output 
	y_output = y_output.flatten()
	val = sum(y-y_output ==0.0)/len(y)

	total += sum(y-y_output ==0.0)
	denom +=len(y)
	# print(val*100)
	# print(y,y_output)
	return val*100


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
	new_data = np.append(X,np.expand_dims(new_Y,axis =1 ),axis = 1)
	# print(new_Y)
	if j!= -1:
		new_data = new_data[new_data[:,-1]>0]
	new_data[:,-1] = new_data[:,-1] -1
	return new_data


def k_fold_cross_val(X,Y,k = 3):

	folds_size = int(len(Y)/k)
	indices = np.arange(len(Y))
	np.random.shuffle(indices)
	idx_copy = indices
	split_dataset =  []
	

	for i in range(k):
		fold = []
		j = 0 + i*folds_size
		while(len(fold)<folds_size):

			fold.append(idx_copy[j])
			j+=1
		split_dataset.append(fold)

	# print(split_dataset)

	return split_dataset



def OnevAll(data):
	global total,denom


	X,Y = preprocess(data)

	indices = k_fold_cross_val(X, Y,5)
	
	specs = np.zeros(3)
	for ii,idxs in enumerate(indices):
		
		print('---------')
		print('fold',ii + 1)
		W =[]
		temp = []
		for rr, idxx in enumerate(indices):
			if rr != ii:
				temp.append(idxx)
		temp = sum(temp,[])
		
		X_train = X[temp]
		y_train = Y[temp]
		X_test = X[idxs]
		y_test = Y[idxs]
	
		for i in set(Y):

			new_data = data_segment(X_train, y_train, i)
			w = lr.logistic_regr(new_data[:,:-1],new_data[:,-1], alpha = 0.01)
			W.append(w)

		temp =np.zeros([len(y_test),3])
		for xx,w in enumerate(W):
			y_output = sigmoid(np.dot(X_test,w))
			temp[:,xx] = y_output.flatten()
		result = np.argmax(temp,axis = 1)
		
		cc = np.zeros([3,3])
		for yy,yyy in zip(result,y_test):
			cc[int(yy)][int(yyy)] +=1
		print(cc/np.sum(cc,axis=1))
		

	print(specs/5)


		
def OnevOne (data):
		
	global total, denom
	specs = np.zeros(3)
	X,Y = preprocess(data)
	indices = k_fold_cross_val(X,Y,5)
	W = []
	
	for ii,idxs in enumerate(indices):
		print('----------')
		print('fold',ii + 1)
		W =[]
		temp = []
		for rr, idxx in enumerate(indices):
			if rr != ii:
				temp.append(idxx)
		temp = sum(temp,[])
		
		X_train = X[temp]
		y_train = Y[temp]
		X_test = X[idxs]
		y_test = Y[idxs]

		for i in range(3):
			for j in range(i+1,3):
				

				new_data = data_segment(X_train, y_train, i,j)
				w = lr.logistic_regr(new_data[:,:-1],new_data[:,-1], alpha = 0.009)
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



	

























	# for i in range(3):

	# 	for j in range(i+1,3):
	# 		print(i,j)
	# 		total = 0.0
	# 		denom = 0.0
	# 		new_data = data_segment(X,Y,i,j)
	# 		X_ ,Y_ = new_data[:,:-2], new_data[:,-1]
	# 		indices = k_fold_cross_val(X_,Y_,3)
	# 		W =[]
	# 		for ii,idxs in enumerate(indices):

	# 			X_test = X_[idxs]
	# 			y_test = Y_[idxs]
	# 			temp = []
	# 			for rr, idxx in enumerate(indices):
	# 				if rr != ii:
	# 					temp.append(idxx)
				
	# 			temp = sum(temp,[])
	# 			print(temp,idxs)
	# 			X_train = X_[temp]
	# 			y_train = Y_[temp]
	# 			data_part = {
	# 						'X_train' : X_train,
	# 						'y_train' : np.expand_dims(y_train, axis = 1),
	# 						'X_test' : X_test,
	# 						'y_test' : y_test
	# 						}

	# 			# print(data_part)


	# 			w,_,__ = Ques_7.logistic_regr(dataa = data_part,alpha = 0.001,flag = 1)
	# 			print('fold ',ii+1)
	# 			W.append(w)
	# 			measure_specs(X_test, y_test, w)
			
			# print('acc for ',i,j,'is ')
			# print(total/denom*100)
	# print('total acc is: ')
	# print(total/denom *100)






data = pd.read_excel('data4.xlsx',header = None)
data =np.array(data)
# print(data)
OnevOne(data)