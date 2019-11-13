import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#===================================
# np.random.seed(4)

def k_means(data,clusters,epochs = 50):

	
	centers_i = np.random.choice(np.shape(data)[0],clusters)
	centers = data[centers_i]
	# centers = np.random.rand(clusters, np.shape(data)[1])
	# print(centers)
	dists = np.zeros([len(data),clusters])
	cluster = np.zeros(len(data))

	for ep in range(epochs):
		for idx1,point in enumerate(data):

			for idx2,center in enumerate(centers):


				dists[idx1,idx2] = np.linalg.norm(point - center)
			# print(dists[idx1,:])
			cluster[idx1] = np.argmin(dists[idx1,:])
		
		# print(np.shape(cluster))

		for c in range(clusters):
			# print(c)
			clust = np.where(cluster == c)
			# print(np.shape(clust))
			centers[c] = sum(data[clust])/np.shape(clust)[1]
			
		# exit(0)
	return cluster,centers
