import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 50, padding=(0,0))
		self.pool = nn.AvgPool2d(5, 5)
		#self.pool = nn.MaxPool2d(5, 5)
		self.conv2 = nn.Conv2d(6, 16, 10)
		#self.pool2 = nn.Maxpool2d(2,2)
		self.fc1 = nn.Linear(16*2*2, 100)
		self.fc2 = nn.Linear(100, 20)
		self.fc3 = nn.Linear(20, 2)
		self.predict = nn.Linear(2, 1)
	
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		print (x.shape)
		x = x.view(-1, 16*2*2)
		print (x.shape)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		#x = self.fc3(x)
		x = self.predict(x)
		return x


class Net_class(nn.Module):
	def __init__(self):
		super(Net_class, self).__init__()
		#Experiment Note: ss, tn, rrelu dont reduce the loss to the extent that relu does
		self.ss = nn.Softsign()
		self.tn = nn.Tanh()
		self.rrelu = nn.RReLU()
		self.conv1 = nn.Conv2d(1, 15, 50, padding=(0,0))
		self.pool = nn.AvgPool2d(5, 5)
		#self.pool = nn.MaxPool2d(5,5)
		self.conv2 = nn.Conv2d(15, 30, 10)
		#self.pool2 = nn.Maxpool2d(2,2)
		self.fc1 = nn.Linear(30*2*2, 100)
		self.fc2 = nn.Linear(100, 20)
		#self.fc3 = nn.Linear(100, 50)
		#self.predict = nn.Linear(20, 2)
		self.predict = nn.Linear(20, 1)
		self.sig = nn.Sigmoid()
	
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		print (x.shape)
		x = x.view(-1, 30*2*2)
		#print (x.shape)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		#x = F.relu(self.fc3(x))
		x = self.predict(x)
		x = self.sig(x)
		return x


class Net_class_linear(nn.Module):
	def __init__(self):
		super(Net_class_linear, self).__init__()
		self.fc1 = nn.Linear(2, 20)
		#self.fc2 = nn.Linear(200, 100)
		#self.fc3 = nn.Linear(100, 50)
		self.predict = nn.Linear(20, 1)
	
	def forward(self, x):
		x = x.view(-1, 2)
		print (x.shape)
		x = F.relu(self.fc1(x))
		#x = F.relu(self.fc2(x))
		#x = F.relu(self.fc3(x))
		#x = self.fc3(x)
		x = self.predict(x)
		return x


class Net_translational(nn.Module):
	def __init__(self):
		super(Net_translational, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, kernel_size=(15,150), padding=(0,0))
		self.pool = nn.MaxPool2d(5, 5)
		self.conv2 = nn.Conv2d(6, 16, 10)
		#self.pool2 = nn.Maxpool2d(2,2)
		self.fc1 = nn.Linear(6*136, 20)
		self.fc2 = nn.Linear(200, 100)
		self.fc3 = nn.Linear(100, 50)
		self.predict = nn.Linear(20, 1)
	
	def forward(self, x):
		#x = self.pool(F.relu(self.conv1(x)))
		x = F.relu(self.conv1(x))
		#x = self.pool(F.relu(self.conv2(x)))
		print (x.shape)
		x = x.view(-1, 6*136)
		print (x.shape)
		x = F.relu(self.fc1(x))
		#x = F.relu(self.fc2(x))
		#x = F.relu(self.fc3(x))
		#x = self.fc3(x)
		x = self.predict(x)
		return x


class Net_translational_patch(nn.Module):
	def __init__(self):
		super(Net_translational_patch, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, kernel_size=(3,15), padding=(0,0))
		self.pool = nn.MaxPool2d(5, 5)
		self.conv2 = nn.Conv2d(6, 16, 10)
		#self.pool2 = nn.Maxpool2d(2,2)
		self.fc1 = nn.Linear(6*136, 20)
		self.fc2 = nn.Linear(200, 100)
		self.fc3 = nn.Linear(100, 50)
		self.predict = nn.Linear(20, 1)
	
	def forward(self, x):
		#x = self.pool(F.relu(self.conv1(x)))
		x = F.relu(self.conv1(x))
		#x = self.pool(F.relu(self.conv2(x)))
		print (x.shape)
		x = x.view(-1, 6*136)
		print (x.shape)
		x = F.relu(self.fc1(x))
		#x = F.relu(self.fc2(x))
		#x = F.relu(self.fc3(x))
		#x = self.fc3(x)
		x = self.predict(x)
		return x


class Net_linear(nn.Module):
	def __init__(self):
		super(Net_linear, self).__init__()
		#takes (150, 150)
		self.fc1 = nn.Linear(150*150, 200)
		self.fc2 = nn.Linear(200, 120)
		self.fc3 = nn.Linear(120, 60)
		self.predict = nn.Linear(120, 1)
	
	def forward(self, x):
		x = x.view(-1, 150*150)
		#print (x.shape)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		#x = F.relu(self.fc3(x))
		x = self.predict(x)
		return x


class Net_linear_sum(nn.Module):
	def __init__(self):
		super(Net_linear_sum, self).__init__()
		#takes (286, 301)
		self.fc1 = nn.Linear(286, 200)
		self.fc2 = nn.Linear(200, 120)
		self.fc3 = nn.Linear(120, 60)
		self.predict = nn.Linear(60, 1)
	
	def forward(self, x):
		x = x.view(-1, 286)
		#print (x.shape)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.predict(x)
		return x
