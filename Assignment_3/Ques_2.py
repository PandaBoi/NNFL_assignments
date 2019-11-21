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
# np.random.seed(69)
#===========================================================================================

class ECG_dataset(Dataset):

	def __init__(self,X,Y):

		self.X = X
		self.Y = Y

	def __len__(self):
		return len(self.Y)

	def __getitem__(self,idx):

		X = np.reshape(self.X[idx],(1,1000))
		return torch.from_numpy(X),torch.from_numpy(X)


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()


		self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 8, kernel_size = 10,stride = 1)
		# self.drop1 = nn.Dropout(p =0.6)
		self.pool1 = nn.MaxPool1d(4,return_indices = True)
		
		self.fc1 = nn.Linear(1976,1976)
		self.up = nn.MaxUnpool1d(4)
		self.conv11 = nn.ConvTranspose1d(in_channels = 8,out_channels = 1,kernel_size = 11,stride = 1)

	def forward(self,x):

		x = x.view(x.size(0),1,-1)

		x = F.relu(self.conv1(x))
		# x = self.drop1(x)
		# x = x.view(x.size(0),-1)
		# print(np.shape(x))
		x,indices = self.pool1(x)
		x = x.view(x.size(0),-1)

		# print(np.shape(x))
		x = F.relu(self.fc1(x))
		# print(np.shape(x))
		# x = self.drop1(x)
		# x = x.view(x.size(0),1,-1)
		x = x.view(x.size(0),8,-1)
		# print(np.shape(x))
		x = self.up(x,indices)
		# x = self.drop1(x)
		x = self.conv11(x)

		return x.squeeze()


model = Net().to(device)
summary(model,(1,1,1000))










X = scio.loadmat('data_for_cnn.mat')
Y = scio.loadmat('class_label.mat')

X = X['ecg_in_window']

X = np.transpose(X/np.max(X))
Y = Y['label']

# print(np.shape(X),np.shape(Y))

ecg_data = ECG_dataset(X, Y)

#generating indexes at random for train validation split
validation_part = 0.2
num_train = len(X)
ind = list(range(num_train))
np.random.shuffle(ind)
split = int(np.floor(validation_part*num_train))
train_idx ,valid_idx = ind[split:] , ind[:split]

#defining sampler for training and validation sets
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

batches = 2
train_load = DataLoader(ecg_data, batch_size = batches, sampler = train_sampler, num_workers = 0)
valid_load = DataLoader(ecg_data, batch_size = batches, sampler = valid_sampler, num_workers = 0)





criterion = nn.MSELoss()
optimizer  = torch.optim.Adam(model.parameters() , lr =0.001)

epochs = 5
valid_loss_min = np.Inf
prev_acc = 0

train_losss = []
valid_losss = []
#----------------------------------------------------------------------------------
for epoch in range(epochs):
	train_loss = 0.0
	valid_loss = 0.0

	#training -------------------------------------------------------------------
	model.train()
	for data, label in train_load:
		# print(data,label)
		data  = data.to(device)
		data = data.float()
		data = data.squeeze()
		# label = label.type(torch.FloatTensor)
		# print(label)
		

		optimizer.zero_grad()

		logps = model(data)
		# print(np.shape(logps),np.shape(data))
		
		loss = criterion(logps,data)
		loss.backward()
		
		optimizer.step()
		
		train_loss += np.sqrt(loss.item()*data.size(0))

# 	# validation----------------------------------------------------------------------
	

	model.eval()
	labels = []
	correct_count = 0.0
	total_count = 0.0
	for data,label in valid_load:

		data  = data.to(device)
		data = data.float()
		data = data.squeeze()

		output = model(data)
		# print(np.shape(output),np.shape(data))
		loss = criterion(output,data)
		valid_loss += np.sqrt(loss.item()*data.size(0))
	# 	output = output.view(-1)
	# 	output[output>=0.5] = 1
	# 	output[output<0.5] = 0
	# 	label = label.view(-1)
	# 	correct_count += (output == label).double().sum().item()
	# 	total_count += output.size(0)
	# 	# print(1*(output.item()>0.5),label.squeeze())
	# print(correct_count/len(valid_idx))

	print('epoch',epoch+1,'train:',train_loss/len(train_load.dataset),'valid',valid_loss/len(valid_load.dataset))
	train_losss.append(train_loss)
	valid_losss.append(valid_loss)


# plt.plot(train_losss)
# plt.plot(valid_losss)
# plt.show()

model.eval().cpu()

dataiter = iter(valid_load)

x,y = dataiter.next()
x = x.squeeze().float()
# print(x,y)
output = model(x)
output = np.array(output.detach()).reshape(batches,-1)
print(np.shape(output))
x = np.array(x).reshape(batches,-1)

plt.figure()
plt.plot(output[0])
plt.figure()
plt.plot(x[0])

plt.show()