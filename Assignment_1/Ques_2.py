import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BATCH = 1
#======================

def preprocess(data):

	x1 = data[:,[0]]
	x2 = data[:,[1]]
	y = data[:,[2]]
	
	x1_mean =np.mean(x1)
	x2_mean =np.mean(x2)
	y_mean =np.mean(y)
	x1_std = np.std(x1)
	x2_std = np.std(x2)
	y_std = np.std(y)


	x1 = (x1 - x1_mean)/x1_std
	x2 = (x2 - x2_mean)/x2_std
	y = (y - y_mean)/y_std

	bias = np.expand_dims(np.ones([len(x1)]),axis = 1)
	X = np.append(bias,x1,axis = 1)
	X = np.append(X,x2,axis = 1)
	# print(np.shape(X),np.shape(y))

	return X, y


def del_j (h_x,X,y,W,idx):

	del_j = 0
	# print(X)
	indices = np.random.choice(len(y),BATCH)

	for i in indices:
		del_j += (h_x[i] - y[i])*X[i,idx] 

	return del_j

def compute_J(h_x,y):

	m = np.shape(h_x)[0]
	# print(m)
	temp = h_x - y
	# print(temp**2)
	J = (1/(2*m)) * (np.sum(temp**2))
	# print(J)

	return J

def stoch_grad(data,alpha = 2e-4,epsilon = 1e-10):

	X, y = preprocess(data)
	cost_value = []
	W = np.zeros([3,1])
	w1 = []
	w2 = []
	eps = np.Inf
	# X = data[:,[0,1]]
	# bias = np.expand_dims(np.ones([len(X)]),axis = 1)
	
	# y = data[:,[2]]
	# X = np.append(bias,X,axis =1)
	epochs = 1000
	ep =0
	h_x = np.dot(X,W)
	# print(np.shape(X),np.shape(W))
	

	indices = np.random.choice(len(X),len(X))

	while(eps>epsilon) and (ep < epochs):
		
	
		for j in range(len(W)):
			W[j] = W[j] - alpha*(del_j(h_x, X, y, W, j))
			

		h_x = np.dot(X,W)
		w1.append(W[1,0])
		w2.append(W[2,0])
		cost_value.append(compute_J(h_x,y))
		ep +=1
		
		if(len(cost_value)>1):
			eps = np.abs(cost_value[-1] - cost_value[-2])
			# print(eps)

	return cost_value,W, w1, w2


data = pd.read_excel('data.xlsx',header = None)
data = np.array(data)
ALPHA = 0.004

values, w,w1,w2= stoch_grad(data,ALPHA)
print('W values ', w)


plt.plot(values)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')


ax.plot(w1,w2,values)
# ax.plot_trisurf(w1,w2,values)
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('Cost function')
plt.show()


fig2 = plt.figure()
ax = fig2.add_subplot(111,projection = '3d')
ww1 = np.linspace(min(w1)-0.05,max(w1)+0.05,1000)
ww2 = np.linspace(min(w2)-0.05,max(w2)+0.05,1000)
W = np.zeros([3,1])
J_cont = np.zeros([len(ww1),len(ww2)])
X,Y = preprocess(data)
for i1,w_1 in enumerate(ww1):
	for i2,w_2 in enumerate(ww2):
		W[0] = w[0]
		W[1] = w_1
		W[2] = w_2
		# print(np.shape(X),np.shape(W))
		h_x = np.dot(X,W)
		# print(h_x)

		J_cont[i1][i2] = compute_J(h_x, Y)

v = np.squeeze(values)
www1,www2 = np.meshgrid(ww1,ww2)
www_1,www_2 = np.squeeze(ww1), np.squeeze(ww2)
plt.plot(w1,w2,values,'r--',zorder = 10)
# plt.contourf(www1,www2,J_cont)
ax.plot_surface(www1,www2,J_cont,alpha = 0.7)
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('Cost function')
plt.show()

