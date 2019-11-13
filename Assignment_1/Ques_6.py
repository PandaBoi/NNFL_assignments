import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#===================================

def preprocess(data):
	print(np.shape(data))
	for i in range(np.shape(data)[-1]):

		temp_mean = np.mean(data[:,i])
		temp_std = np.std(data[:,i])
		data[:,i] = (data[:,i] - temp_mean)/temp_std

	return data

def k_means(data,clusters = 2,epochs = 500):

	data = preprocess(data)

	centers = np.random.rand(clusters, np.shape(data)[1])
	print(centers)
	dists = np.zeros([len(data),clusters])
	cluster = np.zeros(len(data))

	for ep in range(epochs):
		for idx1,point in enumerate(data):

			for idx2,center in enumerate(centers):


				dists[idx1,idx2] = np.linalg.norm(point - center)
			cluster[idx1] = np.argmin(dists[idx1,:])
		print(cluster)

		for c in range(clusters):

			clust = np.where(cluster == c)

			centers[c] = sum(data[clust])/len(clust)
		# print(centers)	

	return cluster





data = pd.read_excel('data2.xlsx',header = None)
data = np.array(data)
clusters = k_means(data)

c1 = np.where(clusters ==0)
c2= np.where(clusters ==1)

# plt.scatter(data[c1,0], data[c1,1],)
# plt.scatter(data[c2,2],data[c2,3])
fig = plt.figure()
fig.subplots(4, 1)

for i in range(4):
	a = np.zeros(np.shape(c1))
	b = np.ones(np.shape(c2))
	fig.add_subplot(4,1,i+1)	
	plt.scatter(data[c1,i],a)
	plt.scatter(data[c2,i],b)
	plt.xlabel('feature')
	plt.ylabel('label')
	
plt.show()
# print(data)
