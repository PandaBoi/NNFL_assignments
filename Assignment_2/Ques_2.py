import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import k_means as km
from sklearn.cluster import KMeans
#================================================
np.random.seed(69)
def preprocess(data):
	for i in range(np.shape(data)[1] -1):
		
		data[:,i] = (data[:,i] - data[:,i].mean())/data[:,i].std()

	X = data[:,:-1]
	Y = data[:,-1]
	# print(np.shape(X))
	
	return X,Y


def test_train_split(X, y,p = 0.7):
	
	indices = np.arange(len(y))
	np.random.shuffle(indices)
	slice_portion = int(p*len(y))
	
	train_slice, test_slice = indices[:slice_portion], indices[slice_portion+1:]
	X_train, X_test, y_train, y_test = X[train_slice], X[test_slice], y[train_slice], y[test_slice]
	# print(X_test,y_test)
	return X_train, X_test, y_train, y_test

def calc_params(X,labels,mu,k):

	indices = dict()
	sigmas =[]
	
	labelz = list(set(labels))
	# print(labelz)
	for l in labelz:
		indices[l] = np.where(labels == l)
	# print(indices)

	for l in labelz:
		m = np.shape(indices[l])[1]
		# print(l,indices[l])
		val =0
		ind = indices[l][0]
		for ii in ind:
			# print(np.linalg.norm(X[ii]-mu[int(l)]))

			val += np.linalg.norm(np.around(X[ii],17) -mu[int(l)])

			if  val == 0 and m == 1:
				print(X[ii][13],mu[int(l)][13])


		sigmas.append(val/m)
	# print(sigmas)

	sigmas = np.array(sigmas)

	betas = 1/(2*(sigmas)**2)
	# print(betas)



	return sigmas,betas


def radial(x,mu,beta):
	return np.exp(-beta*(np.linalg.norm(x - mu)**2))

def linear(x,mu,beta):
	return np.linalg.norm(x - mu)

def multi_quad(x,mu,sigma):
	return (np.linalg.norm(x-mu)**2 + sigma**2)**0.5


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


def RBFN(X_train,X_test,y_train,y_test,k = 10):

	# X_train, X_test, y_train, y_test = test_train_split(X, Y,p = 0.7)
	Y = np.zeros([np.shape(X_train)[0],2])
	
	for i in range(np.shape(X_train)[0]):
		if y_train[i] == 0:
			Y[i] = [1,0]
		else:
			Y[i] = [0,1]
	# print(Y)

	# clusters, centers = km.k_means(X_train,clusters = k)
	kmeans = KMeans(n_clusters = k)
	kmeans.fit_predict(X_train)
	clusters = kmeans.predict(X_train)
	centers = kmeans.cluster_centers_

	mu = centers
	# print(clusters)
	# print(np.shape(centers))
	sigmas,betas = calc_params(X_train,clusters,centers,k)
	# print(betas)
	# print()
	# print(mu)
	# print()
	H = np.zeros([np.shape(X_train)[0],k])
	for i,m in enumerate(X_train):
		for kk in range(k):
			# H[i][kk] = radial(X_train[i], mu[kk], betas[kk])
			H[i][kk] = linear(X_train[i], mu[kk], sigmas[kk])
			# H[i][kk] = multi_quad(X_train[i], mu[kk], sigmas[kk])

	

	# print(H)
	W = np.dot(np.linalg.pinv(H),Y)
	# print()

	H1 = np.zeros([np.shape(X_test)[0],k])
	for i,m in enumerate(X_test):
		for kk in range(k):
			# H1[i][kk] = radial(X_test[i], mu[kk], betas[kk])
			H1[i][kk] = linear(X_test[i], mu[kk], sigmas[kk])
			# H1[i][kk] = multi_quad(X_test[i], mu[kk], sigmas[kk])

	# print(H1)

	

	Y_out = np.dot(H1,W)
	# print(Y_out)
	y_pred = []

	for y in Y_out:
		y_pred.append(np.argmax(y))
	# print(y_pred)


	cc = np.zeros([2,2])
	for yy,yyy in zip(y_pred,y_test):
		# print(yy,yyy)
		cc[int(yy)][int(yyy)] +=1
	print(cc)
	print('acc',(cc[0][0]+cc[1][1])/sum(sum(cc)))
	print('sens',(cc[0][0])/(cc[0][0]+cc[1][0]))
	print('spec',(cc[1][1])/(cc[1][1]+cc[0][1]))

	
	
	













data = sio.loadmat('data5.mat')
data = data['x']
X,Y = preprocess(data)

# W = RBFN(X,Y,k = 300)


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


		W = RBFN(X_train,X_test,y_train,y_test,k = 550)










