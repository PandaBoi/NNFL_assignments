import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
#=======================================================
np.random.seed(69)




def preprocess(data):

	for i in range(np.shape(data)[1] - 1):

		data[:,i] = (data[:,i] - data[:,i].mean())/data[:,i].std()

	X = data[:,:-1]
	Y = data[:,-1]
	bias = np.ones([len(Y),1])
	X = np.append(bias,X,axis = 1)
	
	
	return X,Y




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



def gaussian_v(x,L,train = 1):

	global a,b
	
	if train ==1:
		print('gen')
		a = np.random.rand(np.shape(x)[1],L) 
		b = np.random.rand(L)

	# print(np.shape(a))
	H = np.zeros([np.shape(x)[0],L])

	for i in range(np.shape(H)[0]):
		for j in range(np.shape(H)[1]):
			# print(x[i]-a[:,j])
			H[i][j] = np.exp(-b[j]*np.linalg.norm(x[i]-a[:,j]))
	# print(np.shape(H))
	return H*1.5 + 2


def tanh_v(x,L,train = 1):
	global a,b
	if train ==1:
		print('gen')
		a = np.random.randn(np.shape(x)[1],L) 
		b = np.random.randn(L)

	H = np.zeros([np.shape(x)[0],L])

	for i in range(np.shape(H)[0]):
		for j in range(np.shape(H)[1]):
			prod = np.dot(np.transpose(x[i]),a[:,j])
			# print(prod,'/')
			H[i][j] = (1 - np.exp(-(prod+b[j])))/(1 + np.exp(-(prod+b[j])))
	
	
	return H*1.5 +1

def ELM(X_train,Y_train,X_test,y_test,L = 500): #gaussian 500

	y = np.zeros([np.shape(X_train)[0],2])

	for i in range(np.shape(X_train)[0]):
		if Y_train[i] == 0:
			y[i] = [1,0]
		else:
			y[i] = [0,1]

	# H = np.zeros([np.shape(X_train)[0],L])

	# for i in range(np.shape(H)[0]):
	# 	for j in range(np.shape(H)[1]):

	# H = gaussian_v(X_train,L)
	H = tanh_v(X_train,L)

			
	# print(np.shape(H),H)

	W = np.dot(np.linalg.pinv(H),y)
	# print(np.shape(W))
	# H1 = gaussian_v(X_test, L,train = 0)
	H1 = tanh_v(X_test,L,train = 0)

	# H1 = np.zeros([np.shape(X_test)[0],L])

	# for i in range(np.shape(H1)[0]):
	# 	for j in range(np.shape(H1)[1]):

	# 		H1[i][j] = gaussian(X_test[i])
	# 		# H1[i][j] = tanh_v(X_test[i])

	output = np.dot(H1,W)
	y_pred = []
	for o in output:
		y_pred.append(np.argmax(o))
	y_pred = np.array(y_pred)
	# print(y_pred)

	# print(np.shape(np.where((y_pred - y_test)==0))[1]/len(y_test))
	cc = np.zeros([2,2])
	for yy,yyy in zip(y_pred,y_test):
		# print(yy,yyy)
		cc[int(yy)][int(yyy)] +=1
	print(cc)
	print('acc',(cc[0][0]+cc[1][1])/sum(sum(cc)))
	print('sens',(cc[0][0])/(cc[0][0]+cc[1][0]))
	print('spec',(cc[1][1])/(cc[1][1]+cc[0][1]))

	

	return np.shape(np.where((y_pred - y_test)==0))[1]/len(y_test)
















data = sio.loadmat('data5.mat')
data= data['x']
X,Y = preprocess(data)
val = 0.0
indices = k_fold_cross_val(X, Y, k = 5)
for ii,idxs in enumerate(indices):
		
		print('---------')
		print('fold',ii + 1)
	
		temp = []
		for rr, idxx in enumerate(indices):
			if rr != ii:
				temp.append(idxx)
		temp = sum(temp,[])
		
		X_train = X[temp]
		y_train = Y[temp]
		X_test = X[idxs]
		y_test = Y[idxs]

		val += ELM(X_train,y_train,X_test,y_test)
print('overall accuracy')
print(val/5)