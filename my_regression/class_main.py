import argparse
from matplotlib			import pyplot as plt
import numpy as np
import os
from PIL	import Image
import sys
from scipy.io import FortranFile
import torch
from torch.autograd		import Variable


from model 				import Net, Net_linear, Net_translational,\
								Net_class, Net_class_linear

sys.path.append('/home/patel.3140/python_my_utils/')
from my_utils			import writable

#glo_min,glo_max = 1e-4, 1e-1

#this is for re70k_fsi
def read_input_rescale(dir_nm, num_mode, fl_nm):
	f = FortranFile("../ML/%s/%s.q"%(dir_nm, fl_nm))
	f.read_ints()
	x,_,y,tot_num_mode = f.read_ints()
	X = f.read_reals()
	#first 3 dimension are shape of grid. last is the number of modes
	#grids are of shape 286,1,301
	#fortran format - first index changing fastest
	X = X.reshape(x, y, tot_num_mode, order='F')
	#1 is for channel
	X = X.reshape(1, x, y, tot_num_mode)
	X = X[:,:,:,0:num_mode]
	X = np.moveaxis(X, 3, 0)
	#only consider 76-226 for vertical grid which is necessary
	# and 76:676 for horizontal
	X = X[:,:,76:226,76:676]
	X_new = np.zeros((X.shape[0], 1, 150, 150))
	#normalize and rescale
	for i in range(X.shape[0]):
		img = Image.fromarray(X[i][0])
		img_rescaled = img.resize((150,150))
		X_new[i][0] = np.asarray(img_rescaled)
	for i in range(X.shape[0]):
		X_new[i] = (X_new[i]-X_new[i].mean())/X_new[i].std()

	return X_new


def read_input(dir_nm, num_mode, fl_nm, is_normalize=True):
	f = FortranFile("../ML/%s/%s.q"%(dir_nm, fl_nm))
	f.read_ints()
	f.read_ints()
	X = f.read_reals()
	#first 3 dimension are shape of grid. last is the number of modes
	#grids are of shape 286,1,301
	#fortran format - first index changing fastest
	X = X.reshape(286, 301, 10, order='F')
	X = X.reshape(1, 286, 301, 10)
	#1 is for channel
	X = X[:,:,:,0:num_mode]
	X = np.moveaxis(X, 3, 0)
	if (fl_nm == "plate_1"):
		xlow,ylow,xhigh,yhigh=76,76,226,226 
	elif (fl_nm == "plate_2"):
		xlow,ylow,xhigh,yhigh=76,132,226,282 
	else:
		print ("not known file")
	#only consider 76-226 grid which is necessary
	X = X[:,:,ylow:yhigh,xlow:xhigh]
	#normalize
	if (is_normalize):
		for i in range(X.shape[0]):
			X[i] = (X[i]-X[i].mean())/X[i].std()

	return X


def train_with_MSELoss(x, y, net, tot_epochs):
	optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
	#loss_func = torch.nn.CrossEntropyLoss()
	loss_func = torch.nn.MSELoss()
	#loss_func = torch.nn.SmoothL1Loss()
	
	for t in range(tot_epochs):
		prediction = net(x)
		print (prediction.shape)
		prediction = prediction.reshape(y.shape[0])
		loss = loss_func(prediction, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print ("%d loss: %f"%(i, loss.data.numpy()))
		if (t==tot_epochs-1):
			for i in range(y.shape[0]):
				print (i+1)
				print ("Ground Truth %.3f"% y[i])
				print ("Prediction %.3f"%(prediction[i]))
	return prediction


def train_with_BinaryEntropLoss(x, y, x_test, y_test, net, tot_epochs):
	optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
	loss_func = torch.nn.BCELoss()
	
	for t in range(tot_epochs):
		prediction = net(x)
		loss = loss_func(prediction, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print ("%d loss: %f"%(t, loss.data.numpy()))
		#if (t==tot_epochs-1):
			#for i in range(y.shape[0]):
				#print (i+1)
				#print ("Ground Truth %.3f"% y[i])
				#print ("Prediction %.3f"%(prediction[i]))
	#estimate the scaling parameters from the train set
	output = net(x).data.numpy()
	#output = norm(output, "unit")
	
	t_prediction = net(x_test).reshape(-1, 1)	
	prediction = t_prediction.data.numpy()
	#prediction = norm(prediction, "unit")
	print (prediction)
	print (y_test)
	#cnt_corr = (predicted==y_test).sum().item()
	#print ("Accuracy %.3f"%(100*cnt_corr/float(y_test.shape[0])))
	print ("Test Loss %f\n"% loss_func(t_prediction, y_test.reshape(-1,1)))
	return prediction, output


def train_with_CrossEntropLoss(x, y, x_test, y_test, net, tot_epochs):
	optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
	loss_func = torch.nn.CrossEntropyLoss()
	#loss_func = torch.nn.BCELoss()
	#loss_func = torch.nn.MSELoss()
	#loss_func = torch.nn.SmoothL1Loss()
	
	for t in range(tot_epochs):
		prediction = net(x)
		loss = loss_func(prediction, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print ("loss: %f"%loss.data.numpy())
		if (t==tot_epochs-1):
			for i in range(y.shape[0]):
				print (i+1)
				print ("Ground Truth %.3f"% y[i])
				print ("Prediction %.3f, %.3f"%(prediction[i][0],prediction[i][1]))
	output = net(x_test)
	_, predicted = torch.max(output, 1)	
	cnt_corr = (predicted==y_test).sum().item()
	print ("Accuracy %.3f"%(100*cnt_corr/float(y_test.shape[0])))
	return prediction


def make_plots(x, fl_nm):
	import pandas as pd
	import seaborn as sns
	plt.rcParams['figure.figsize'] = (20.0, 10.0)
	plt.rcParams['font.family'] = "serif"
	df = pd.DataFrame(x)
	df.reindex()
	ax = sns.heatmap(df, cmap='Greys')
	sns.set(font_scale=4)
	#ax.set_yticklabels([])
	#ax.set_xticklabels([])
	#ax.invert_yaxis()
	plt.savefig(writable("plots",fl_nm+".png"), bbox_inches='tight')
	plt.close()


def numerical_to_class(Y):
	Y_class = np.zeros(Y.shape[0],dtype=np.long)
	for i in range(Y.shape[0]):
		if (Y[i]>0): 
			Y_class[i]=0
		else: 
			Y_class[i]=1
	Y_class=Y_class.astype(dtype=int)
	return Y_class


def norm(x, methd="std"):
	if methd=="std":
		return (x - x.mean())/x.std()
	elif methd=="id":
		return x
	elif methd=="unit":
		return (x-x.min())/(x.max()-x.min())
	elif methd=="unit_ab":
		assert(x.min()>glo_min)
		assert(x.max()<glo_max)
		return (x-glo_min)/(glo_max-glo_min)
	elif methd=="sigmoid":
		return 1/(1+np.exp(-x))


def augment_data(X, Y):
	new_X = np.zeros(X.shape)
	new_Y = np.zeros(Y.shape)
	#flip 180
	for i in range(X.shape[0]):
		new_X[i] = np.fliplr(X[i])
		new_Y[i] = Y[i]
	#new_X = new_X.reshape	
	X = np.concatenate((X, new_X), axis=0)
	Y = np.concatenate((Y, new_Y), axis=0)
	return X, Y


def main(opts):
	#col = 1 for frequency
	#col = 0 for amplitude
	#col = 2 for phase
	global glo_min 
	global glo_max 
	if (opts.char_val == "ampt"):
		col = 0
		glo_min, glo_max = 0.001, 100 
	elif (opts.char_val == "freq"):
		col = 1
		glo_min, glo_max = 0.01, 1e-1  
	elif (opts.char_val == "phase"):
		col = 2
		glo_min, glo_max = -np.pi, np.pi
	
	#number of modes to use for training and testing
	train_num_mode = 10 
	test_num_mode = 10
	tot_epochs = opts.epoch
	lst_dir_nm = opts.train_dir_nm.split(",")
	test_dir_nm = opts.test_dir_nm
	if len(lst_dir_nm)==2:
		#first 
		X_1 = read_input(lst_dir_nm[0], train_num_mode, opts.fl_nm)
		Y_1 = np.loadtxt("../freq_ampt/my_Results/%s/%s_at.txt"%(lst_dir_nm[0], \
													opts.fl_nm), delimiter=',')
		Y_1 = Y_1[:train_num_mode,col]
		#second
		X_2 = read_input(lst_dir_nm[1], train_num_mode, opts.fl_nm)
		Y_2 = np.loadtxt("../freq_ampt/my_Results/%s/%s_at.txt"%(lst_dir_nm[1], \
													opts.fl_nm), delimiter=',')
		Y_2 = Y_2[:train_num_mode,col]
	
	#X_3 = read_input(lst_dir_nm[2], train_num_mode, opts.fl_nm)
	#Y_3 = np.loadtxt("../freq_ampt/Results/%s/%s_at.txt"%(lst_dir_nm[2], \
	#												opts.fl_nm), delimiter=',')
		X = np.concatenate((X_1, X_2), axis=0)
		Y = np.concatenate((Y_1, Y_2), axis=0)
	#this is for re70k_fsi
	elif len(lst_dir_nm)==1:
		X = read_input_rescale(lst_dir_nm[0], train_num_mode, opts.fl_nm)
		Y = np.loadtxt("../freq_ampt/my_Results/%s/%s_at.txt"%(lst_dir_nm[0], \
													opts.fl_nm), delimiter=',')
		Y = Y[:train_num_mode,col]
	#X, Y = augment_data(X, Y)

	val_max, val_min = Y.max(), Y.min()
	#X = X_1
	#Y = Y_1
	#Y = norm(Y)
	Y = norm(Y, "unit_ab")
	#Y = norm(Y, "sigmoid")
	if (test_dir_nm != "re70k_fsi"):
		X_test = read_input(test_dir_nm, test_num_mode, opts.fl_nm)
		Y_test = np.loadtxt("../freq_ampt/my_Results/%s/%s_at.txt"%(test_dir_nm, \
													opts.fl_nm), delimiter=',')
	else:
		X_test = read_input_rescale(test_dir_nm, test_num_mode, opts.fl_nm)
		Y_test = np.loadtxt("../freq_ampt/my_Results/%s/%s_at.txt"%(test_dir_nm, \
													opts.fl_nm), delimiter=',')
		
	Y_test = Y_test[:test_num_mode,col]
	#Y_test = norm(Y_test, "")
	Y_test = norm(Y_test, "unit_ab")
	#Y_test = norm(Y_test, "sigmoid")	

	#plots
	for i in range(10):
		make_plots(X[i][0], "%s_%s_mode_%d"%(lst_dir_nm[0], opts.fl_nm, i+1))
	for i in range(10):
		make_plots(X[10+i][0], "%s_%s_mode_%d"%(lst_dir_nm[1], opts.fl_nm, i+1))
	for i in range(10):
		make_plots(X_test[i][0], "%s_%s_mode_%d"%(test_dir_nm, opts.fl_nm, i+1))

	Y_class = numerical_to_class(Y)
	Y_test_class = numerical_to_class(Y_test)
	
	x, y = Variable(torch.from_numpy(X)).float(), Variable(torch.from_numpy(Y)).float()
	x_test, y_test = Variable(torch.from_numpy(X_test)).float(), \
									Variable(torch.from_numpy(Y_test)).float()
	
	#only when you normalize y's to unit normal
	#net = Net_class()
	#class_prediction = train_with_CrossEntropLoss(x, y, x_test, y_test, net, tot_epochs)
	#class_prediction = class_prediction.data.numpy()
	
	#with open("class_prediction_Result", "w") as fl_out:
	#	for i in range(class_prediction.shape[0]):
	#		fl_out.write("%f, %f, %f\n"% ( class_prediction[i][0], class_prediction[i][1], Y[i])) 
	
	net = Net_class()
	pred_test, pred_train = train_with_BinaryEntropLoss(x, y, x_test, y_test, net, tot_epochs)
	#inverse of unit
	#pred_test = pred_test*(val_max-val_min)+val_min
	#pred_train = pred_train*(val_max-val_min)+val_min
	#inverse of sigmoid
	#pred_test = np.log(pred_test/(1-pred_test))
	#pred_train = np.log(pred_train/(1-pred_train))
	#inverse of unit_ab
	pred_test = pred_test*(glo_max-glo_min)+glo_min
	pred_train = pred_train*(glo_max-glo_min)+glo_min

	#freq_amp = "freq" if col else "amp"
		
	with open(writable("Results", "%s_pred_test_%s_%s"%(opts.char_val,test_dir_nm,opts.fl_nm)), "w") as fl_out:
		for i in range(pred_test.shape[0]):
			fl_out.write("%f\n"% (pred_test[i]))

	with open(writable("Results", "%s_pred_train_%s_%s"%(opts.char_val,'_'.join(lst_dir_nm),opts.fl_nm)), "w") as fl_out:
		for i in range(pred_train.shape[0]):
			fl_out.write("%f\n"% (pred_train[i]))

	#check if we can make this working
	#for i in range(class_prediction.shape[0]):
	#	class_prediction[i] = (class_prediction[i]-class_prediction[i].mean())/\
	#													class_prediction[i].std()
	#class_prediction = Variable(torch.from_numpy(class_prediction)).float()
	#y = Variable(torch.from_numpy(Y)).float()
	#net = Net_class_linear()
	#train_with_MSELoss(class_prediction, y, net, tot_epochs)
	
	#net = Net()
	#train_with(x, y, net, tot_epochs)
	#net = Net_linear()
	#train_with(x, y, net, tot_epochs+100)
	#net = Net_translational()
	#train_with(x, y, net, tot_epochs)
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_dir_nm', help='training directory name', required=True)
	parser.add_argument('--test_dir_nm', help='test directory name', required=True)
	parser.add_argument('--char_val', help='char val (freq or ampt or phase)', required=True)
	parser.add_argument('--fl_nm', help='file name', required=True)
	parser.add_argument('--epoch', help='total epochs', type=int, default=20)
	opts = parser.parse_args()
	main(opts)
