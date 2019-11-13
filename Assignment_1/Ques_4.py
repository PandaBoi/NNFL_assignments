import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')
# import Ques_1
# import Ques_2




def preprocess(data):

	x1 = data[:,[0]]
	x2 = data[:,[1]]
	y = data[:,[2]]
	
	# x1_mean =np.mean(x1)
	# x2_mean =np.mean(x2)
	# y_mean =np.mean(y)
	# x1_std = np.std(x1)
	# x2_std = np.std(x2)
	# y_std = np.std(y)



	x1 = (x1 - x1.mean())/x1.std()
	x2 = (x2 - x2.mean())/x2.std()
	y = (y - y.mean())/y.std()

	# print(max(x1))

	bias = np.expand_dims(np.ones([len(x1)]),axis = 1)
	X = np.append(bias,x1,axis = 1)
	X = np.append(X,x2,axis = 1)
	# plt.scatter(x1,y)
	# plt.show()
	# print(np.shape(X),np.shape(y))

	return X, y

def vector_LR(data):

	X, y = preprocess(data)

	W = np.zeros([3,1])
	inv_inside = np.linalg.inv(np.dot(np.transpose(X),X))
	W =  np.dot(np.dot(inv_inside, np.transpose(X)),y)


	return W











data = pd.read_excel('data.xlsx',header = None)
data = np.array(data)

W = vector_LR(data)
print(W)

