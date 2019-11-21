import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
from torchsummary import summary
import torch.nn.functional as F
use_gpu = torch.cuda.is_available()

if use_gpu:
	print("Using CUDA")
	device =torch.device("cuda:0")
else:
	print("CPU")
	device = torch.device("cpu")
np.random.seed(69)
#===========================================================================================

class ECG_dataset(Dataset):

	def __init__(self,X,Y):

		self.X = X
		self.Y = Y

	def __len__(self):
		return len(self.Y)

	def __getitem__(self,idx):

		return self.X[idx], self.Y[idx]



class CNN_1D(nn.Module):

	def __init__(self):
		super(CNN_1D, self).__init__()

		self.conv_layers = nn.Sequential (nn.Conv1d(in_channels = 1, out_channels = 128, kernel_size = 20,stride = 1),
										nn.AvgPool1d(kernel_size = 32,stride = 1),
										nn.BatchNorm1d(128),
										nn.ReLU(),
										nn.Conv1d(in_channels = 128, out_channels = 64, kernel_size = 20, stride = 1),
										nn.AvgPool1d(kernel_size = 10),
										nn.BatchNorm1d(64),
										nn.ReLU(),
										)

		self.fc = nn.Sequential(nn.Linear(5952,1024),
								nn.ReLU(),
								nn.Dropout(p = 0.2),
								nn.Linear(1024,512),
								nn.ReLU(),
								# nn.Dropout(p = 0.6),
								nn.Linear(512,64),
								nn.ReLU(),
								nn.Linear(64,1),
								nn.Sigmoid()
								)

	def forward(self,x):


		x = x.view(x.size(0),1,-1)
		# print(np.shape(x))
		x = self.conv_layers(x)
		x = x.view(x.size(0),-1)
		# print(np.shape(x))
		x = self.fc(x)

		
		return x






X = scio.loadmat('data_for_cnn.mat')
Y = scio.loadmat('class_label.mat')

X = X['ecg_in_window']
X = X/np.max(X)
Y = Y['label']

# print(np.shape(X),np.shape(Y))

ecg_data = ECG_dataset(X, Y)

#generating indexes at random for train validation split
validation_part = 0.1
num_train = len(X)
ind = list(range(num_train))
np.random.shuffle(ind)
split = int(np.floor(validation_part*num_train))
train_idx ,valid_idx = ind[split:] , ind[:split]

#defining sampler for training and validation sets
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


train_load = DataLoader(ecg_data, batch_size = 100, sampler = train_sampler, num_workers = 0)
valid_load = DataLoader(ecg_data, batch_size = 100, sampler = valid_sampler, num_workers = 0)


model = CNN_1D()
model.to(device)
# summary(model,(1,1000))

#1e-6
criterion = nn.BCELoss()
optimizer  = torch.optim.Adam(model.parameters() , lr =0.00013)

epochs = 50
valid_loss_min = np.Inf
prev_acc = 0


#----------------------------------------------------------------------------------
for epoch in range(epochs):
	train_loss = 0.0
	valid_loss = 0.0

	#training -------------------------------------------------------------------
	model.train()
	for data, label in train_load:
		# print(data,label)
		data = data.type(torch.FloatTensor)
		label = label.type(torch.FloatTensor)
		# print(label)
		data , label = data.to(device), label.to(device)

		optimizer.zero_grad()

		logps = model(data)
		loss = criterion(logps,label)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()*data.size(0)

# 	# validation----------------------------------------------------------------------
	

	model.eval()
	labels = []
	correct_count = 0.0
	total_count = 0.0
	for data,label in valid_load:

		data = data.type(torch.FloatTensor)
		label = label.type(torch.FloatTensor)
		data , label = data.to(device), label.to(device)

		output = model(data)
		loss = criterion(output,label)
		valid_loss += loss.item()*data.size(0)
		output = output.view(-1)
		output[output>=0.5] = 1
		output[output<0.5] = 0
		label = label.view(-1)
		correct_count += (output == label).double().sum().item()
		total_count += output.size(0)
		# print(1*(output.item()>0.5),label.squeeze())
	print('epoch:',epoch+1,'acc:',correct_count/len(valid_idx))




model.eval()
labels = []
correct_count = 0.0
total_count = 0.0
Y_act = []
pred = []
for data,label in valid_load:

	data = data.type(torch.FloatTensor)
	label = label.type(torch.FloatTensor)
	data , label = data.to(device), label.to(device)

	output = model(data)
	loss = criterion(output,label)
	valid_loss += loss.item()*data.size(0)
	output = output.view(-1)
	output[output>=0.5] = 1
	output[output<0.5] = 0
	label = label.view(-1)
	correct_count += (output == label).double().sum().item()
	Y_act.append(np.array(label.cpu()))
	pred.append(np.array(output.detach().cpu()))
	# total_count += output.size(0)
	# print(1*(output.item()>0.5),label.squeeze())
# print('epoch:',epoch+1,'acc:',correct_count/len(valid_idx))

pred = np.array(pred).reshape(-1)
Y_act = np.array(Y_act).reshape(-1)
cc = np.zeros([2,2])

for i in range(len(Y_act)):

	cc[int(Y_act[i])][int(pred[i])] += 1
print(cc) 


# 	#printing and saving model at min point-------------------------------------------
	

# 	train_loss = train_loss/len(train_load.dataset)
# 	valid_loss = valid_loss/len(valid_load.dataset)
# 	acc = sum(labels)/len(Y[valid_idx])
# 	print(acc)
	
# 	print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
# 		epoch+1, 
# 		train_loss,
# 		valid_loss
# 		))
	
# 	# save model if validation loss has decreased-------------------------------------
	

# 	if  acc > prev_acc:
# 		prev_acc = acc
# 		print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
# 		valid_loss_min,
# 		valid_loss))
# 		torch.save(model.state_dict(), 'model.pt')
# 		valid_loss_min = valid_loss
# # ---------------------------------------------------------------------------------------
# model = CNN_1D().to(device)
# model.load_state_dict(torch.load('model.pt'))
# model.eval()
# labels = []
# correct_count = 0.0
# total_count = 0.0
# for data,label in valid_load:

# 	data = data.type(torch.FloatTensor)
# 	label = label.type(torch.FloatTensor)
# 	data , label = data.to(device), label.to(device)

# 	output = model(data)
# 	output = output.view(-1)
# 	label = label.view(-1)
# 	output[output>=0.5] = 1
# 	output[output<0.5] = 0
# 	correct_count += (output == label).double().sum().item()
# 	total_count += output.size(0)
# 	# labels.append(1*(output.item()>0.5) == label.item())

# print(correct_count/total_count)