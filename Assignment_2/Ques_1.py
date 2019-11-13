import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#================================================
np.random.seed(40)

def sigmoid(temp):

	ans =0.0
	ans = 1 / (1 + np.exp(-temp))

	return ans
	
def sigmoid_del(temp):
	return (np.multiply(temp,(np.ones(np.shape(temp))-temp)))

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


class MLP():

	def __init__ (self,inp_features = 73,out_features = 1,lambdA = 2e-4 ,sizes = [5,3],alpha = 0.01,epochs = 100,thresh = 1e-8):

		self.thresh = thresh
		self.epochs = epochs
		self.alpha =alpha
		self.layers = dict()
		self.weights = dict()
		self.deltas = dict()
		self.activations = dict()
		self.sizes = sizes
		self.output = []
		self.lambdA = lambdA
		self.bias = dict()
		
		#init weights
		self.weights['0_1'] = np.random.rand(inp_features, sizes[0])
		for i,s in enumerate(sizes):

			if s == sizes[-1]:
				self.weights[str(i+1) + '_' + str(i+2)] = np.random.rand(sizes[i], out_features)
			else:	
				self.weights[str(i+1) + '_' + str(i+2)] = np.random.rand(sizes[i], sizes[i+1])

			self.layers['layer'+str(i+1)] = np.zeros([sizes[i]])
			self.deltas['delta'+str(i+1)] = np.zeros([sizes[i]])
			self.activations['act'+str(i+1)] = np.zeros([sizes[i]])
		self.deltas['delta3'] = np.zeros(1)

		# self.bias['b1'] = np.zeros(self.sizes[1])
		# self.bias['b2'] = np.zeros(1)

	
	

	def forward(self,X):
		
		self.output = []
		
		self.bias['b1'] = np.zeros(self.sizes[1])
		self.bias['b2'] = np.zeros(1)
		
		self.layers['layer1'] = np.dot(X,self.weights['0_1']) 
		self.activations['act1'] = sigmoid(self.layers['layer1'])
		self.activations['act2'] = sigmoid(np.dot(self.activations['act1'],self.weights['1_2'] )   + self.bias['b1'] )
		
		self.output.append(sigmoid(np.dot(self.activations['act2'],self.weights['2_3']) ) + self.bias['b2'])	
		self.output = np.array(self.output).flatten()
		# print(self.output)

	def backprop(self,X,Y):

		self.deltas['delta3'] = -np.multiply((Y - self.output),sigmoid_del(self.output))
		# print(np.shape(self.deltas['delta3']))
		# self.deltas['delta3'] = np.expand_dims(self.deltas['delta3'] , axis = 1)
		self.deltas['delta2'] = np.multiply(np.dot(self.deltas['delta3'], np.transpose(self.weights['2_3']) ),sigmoid_del(self.activations['act2']))
		# print(np.shape(self.deltas['delta2']))
		self.deltas['delta1'] =np.multiply(np.dot(self.deltas['delta2'], np.transpose(self.weights['1_2']) ),sigmoid_del(self.activations['act1']))
		# print(np.shape(self.deltas['delta1']))


		for ii,w in enumerate(self.weights):
			

			
				if ii == 0:
					activation = X
				else:
					activation = self.activations['act'+str(ii)]

				
				self.weights[w] = (1-self.alpha*self.lambdA)*self.weights[w] - self.alpha*np.dot(np.expand_dims(activation,1),np.transpose(\
																										np.expand_dims(self.deltas['delta'+str(ii+1)],1)))

		for jj,b in enumerate(self.bias):

			self.bias[b] = self.bias[b] - self.alpha*self.deltas['delta'+str(jj+2)] 

		# print(self.bias[b])





	def train(self,X,Y):
		J = []
		indices = np.arange(np.shape(X)[0])
		np.random.shuffle(indices)
		X,Y = X[indices],Y[indices]
		for e in range(self.epochs):
			if e%100 ==0:	
				print(e)	
			temp = 0.0
			for x,y in zip(X,Y):
				self.forward(x)
				self.backprop(x, y)
				temp += (self.output - y)**2
				# exit(0)
			J.append(temp/len(Y))

			if len(J)>1:
				if abs(J[-1] - J[-2])< self.thresh:
					break
		# print(self.bias)
		# print(self.weights)
		plt.plot(J)
		plt.show()

	def test(self,X,Y):

		self.forward(X)

		output = self.output 
		# print(output)
		output = 1*(output >=0.5)
		# print(output)
		# print(np.shape(Y))
		cc = np.zeros([2,2])
		for yy,yyy in zip(output,Y):
			# print(yy,yyy)
			cc[int(yy)][int(yyy)] +=1
		print(cc)
		print('acc',(cc[0][0]+cc[1][1])/sum(sum(cc)))
		print('sens',(cc[0][0])/(cc[0][0]+cc[1][0]))
		print('spec',(cc[1][1])/(cc[1][1]+cc[0][1]))

		


def preprocess(data):

	for i in range(np.shape(data)[1] - 1):

		data[:,i] = (data[:,i] - data[:,i].mean())/data[:,i].std()

	X = data[:,:-1]
	Y = data[:,-1]
	bias = np.ones([len(Y),1])
	X = np.append(bias,X,axis = 1)
	
	
	return X,Y

def test_train_split(X, y,p = 0.6):
	
	indices = np.arange(len(y))
	np.random.shuffle(indices)
	slice_portion = int(p*len(y))
	
	train_slice, test_slice = indices[:slice_portion], indices[slice_portion+1:]
	X_train, X_test, y_train, y_test = X[train_slice], X[test_slice], y[train_slice], y[test_slice]
	# print(X_test,y_test)
	return X_train, X_test, y_train, y_test



def MLP_run(data,valid = 'hold'):

	X,y = preprocess(data)
	net = MLP(epochs = 500, sizes = [30,20])
	if valid == 'hold':
		X_train, X_test, y_train, y_test = test_train_split(X, y)
		net.train(X_train, y_train)
		net.test(X_test, y_test)
	else: 
		indices = k_fold_cross_val(X, y, k = 5)
		for ii,idxs in enumerate(indices):
				
				print('---------')
				print('fold',ii + 1)
			
				temp = []
				for rr, idxx in enumerate(indices):
					if rr != ii:
						temp.append(idxx)
				temp = sum(temp,[])
				
				X_train = X[temp]
				y_train = y[temp]
				X_test = X[idxs]
				y_test = y[idxs]

				net.train(X_train, y_train)
				net.test(X_test, y_test)
			


	





data = sio.loadmat('data5.mat')
data = data['x']
MLP_run(data,valid = 'hold')
# print(data)