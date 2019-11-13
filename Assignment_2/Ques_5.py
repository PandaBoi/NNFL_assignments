import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
#=======================================================
np.random.seed(40)

def sigmoid(temp):

	ans =0.0
	ans = 1 / (1 + np.exp(-temp))

	return ans
def sigmoid_del(temp):
	return (np.multiply(temp,(np.ones(np.shape(temp))-temp)))


class AutoEncoder_ELM():

	def __init__(self,inp_feat = 73,out_feat = 1000,lambdA = 0.006,sizes = [40,30],alpha = 0.01,pre_ep = 60,epochs = 50,thresh = 1e-8):

		self.thresh = thresh
		self.epochs = epochs
		self.alpha =alpha
		self.elm_layers = out_feat
		self.pre_ep  = pre_ep
		self.layers = dict()
		self.weights = dict()
		self.deltas = dict()
		self.activations = dict()
		self.sizes = sizes
		self.output = []
		self.lambdA = lambdA
		self.bias = dict()


		# initializes weights
		self.weights['0_1'] = np.random.randn(inp_feat, self.sizes[0])
		for i,s in enumerate(sizes):

						
			if s == sizes[-1]:

				self.weights[str(i+1)+'_'+str(i+2)] = np.random.rand(self.sizes[i], out_feat)
			else:
				# print(i)
				self.weights[str(i+1)+'_'+str(i+2)] = np.random.rand(self.sizes[i], self.sizes[i+1])
			self.layers['layer'+str(i+1)] = np.zeros([sizes[i]])
			self.deltas['delta'+str(i+1)] = np.zeros([sizes[i]])
			self.activations['act'+str(i+1)] = np.zeros([sizes[i]])
			try:
				self.bias['b'+str(i+1)] = np.zeros([sizes[i+1]])
			except:
				self.bias['b'+str(i+1)] = np.zeros(out_feat)


		self.deltas['delta'+str(len(sizes)+1)] = np.zeros(1)

		
		# print(self.weights.keys(),self.activations.keys())


	def pre_train(self,X,Y):

		weights_keys = list(self.weights.keys())

		for i,s in enumerate(self.sizes):

			if i ==0:
			
				for e in range(self.pre_ep):
					for x in X:
						self.single_forward(x,self.weights[str(i)+'_'+str(i+1)],i)
						self.single_backprop(x,self.weights[str(i)+'_'+str(i+1)],i)
						# exit(0)
				
				

			else:
				
				for e in range(self.pre_ep):
					# print(e)
					for x in X:
						inp = x
						q = 0
						while(q<i):
							self.single_forward(inp,self.weights[str(q)+'_'+str(q+1)],q)
							inp = self.activations['act'+str(q+1)]
							q+=1
						self.single_forward(self.activations['act'+str(i)],self.weights[str(i)+'_'+str(i+1)],i)
						self.single_backprop(self.activations['act'+str(i)],self.weights[str(i)+'_'+str(i+1)],i)
						# exit(0)

				
				
				
		

	def single_forward(self,X,weights,i):

		self.output = []
		w1 = weights
		w2 = np.transpose(w1)
		

		self.activations['act'+str(i+1)] = sigmoid(np.dot(X,w1))
		
		self.layers['layer'+str(i+1)] = sigmoid(np.dot( self.activations['act'+str(i+1)],w2 )) 



			# print(np.shape(self.layers['layer1']))

	def single_backprop(self,X,weights,i):

		w1 = weights
		w2 = np.transpose(w1)



		

		err = -np.multiply( (X - self.layers['layer'+str(i+1)]),sigmoid_del(self.layers['layer'+str(i+1)]) )
		self.deltas['delta'+str(i+1)] = np.multiply(np.dot(err, np.transpose(w2) ),sigmoid_del(self.activations['act'+str(i+1)])) 

		self.weights[str(i)+'_'+str(i+1)] = (1-self.alpha*self.lambdA)*w1 \
							- self.alpha*np.dot(np.expand_dims(X,1),np.transpose(np.expand_dims(self.deltas['delta'+str(i+1)],1)))




	def forward(self,X,Y):

		self.output =[]
		
		for idx,l in enumerate(self.activations.keys()):
			# print(l)
			if idx == 0:
				# print(np.shape(X),np.shape(self.weights[str(idx)+'_'+str(idx+1)]))
				self.activations[l] = sigmoid(np.dot(X,self.weights[str(idx)+'_'+str(idx+1)]))

			else:
				self.activations[l] = sigmoid(np.dot(self.activations['act'+str(idx)] ,self.weights[str(idx)+'_'+str(idx+1)]) + self.bias['b'+str(idx)])

		
		self.output.append(sigmoid(np.dot(self.activations['act2'],self.weights['2_3']) ) + self.bias['b2'])	
		self.output = np.squeeze(np.array(self.output))









	def gaussian_v(self,x,L,train = 1):

		global a,b
		
		if train ==1:
			print('gen')
			a = np.random.rand(np.shape(x)[1],L) 
			b = np.random.rand(L)

		# print(np.shape(x))
		H = np.zeros([np.shape(x)[0],L])

		for i in range(np.shape(H)[0]):
			for j in range(np.shape(H)[1]):
				# print(x[i]-a[:,j])
				H[i][j] = np.exp(-b[j]*np.linalg.norm(x[i]-a[:,j]))
		# print(np.shape(H))
		return H*1.5 + 1


	def train(self,X,Y):
		
		J =  []
		self.pre_train(X,Y)
		print('pre training done!')

		y = np.zeros([np.shape(X_train)[0],2])

		for i in range(np.shape(X_train)[0]):
			if Y[i] == 0:
				y[i] = [1,0]
			else:
				y[i] = [0,1]

		self.forward(X, Y)
		print(self.output)
		H = self.gaussian_v(self.output,self.elm_layers)

		W = np.dot(np.linalg.pinv(H),y)

		return W
		# print(H)



	def test(self,X,Y,W):

		self.forward(X,Y)

		H = self.gaussian_v(self.output, self.elm_layers, train= -1)

		output = np.dot(H, W)

		y_pred = []
		for o in output:
			y_pred.append(np.argmax(o))
		y_pred = np.array(y_pred)

		y_pred = 1*(y_pred>0.5)
		# print(y_pred)
		print(sum(Y==y_pred)/len(Y))
		
		cc = np.zeros([2,2])
		for yy,yyy in zip(y_pred,y_test):
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







data = sio.loadmat('data5.mat')
data= data['x']
X,Y = preprocess(data)
X_train, X_test, y_train, y_test = test_train_split(X, Y,p =0.7)

net = AutoEncoder_ELM()
W = net.train(X_train,y_train)
net.test(X_test,y_test,W)