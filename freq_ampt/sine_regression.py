import matplotlib.pyplot as plt

import csv
import numpy as np
from scipy.optimize import leastsq

dir_nm='h0.016'
fl_nm='plate_1_at'

np_data = np.loadtxt('../Madhav/%s/%s.txt'%(dir_nm, fl_nm))

for i in range(2, 3):
	x,y=np_data[:,0], np_data[:,i]
	# flag = 0
	# count=0
	# for j,val in enumerate(y):
	# 	if flag == 0:
	# 		if (val<0): count+=1;	
	# 		else: 
	# 			print (count)
	# 			count=1
	# 			flag=1
	# 	else:
	# 		if(val>0): count+=1
	# 		else:
	# 			print (count)
	# 			count=1
	# 			flag=0
	guess_mean = np.mean(y)
	guess_std = 3*np.std(y)/(2**0.5)/(2**0.5)
	guess_phase = 0
	guess_freq = 0.0125*(2*np.pi)
	guess_amp = 0.05
	data_first_guess = guess_std*np.sin(guess_freq*x+guess_phase) + guess_mean
	optimize_func = lambda a: a[0]*np.sin(a[1]*x+a[2]) + a[3] - y
	est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, \
				[guess_amp, guess_freq, guess_phase, guess_mean])[0]
	
	data_fit = est_amp*np.sin(est_freq*x+est_phase) + est_mean
	fine_t = np.arange(0,max(x),0.1)
	data_fit=est_amp*np.sin(est_freq*fine_t+est_phase) + est_mean

	plt.plot(x, y, '.')
	plt.plot(x, data_first_guess, label='first guess')
	plt.plot(fine_t, data_fit, label='after fitting')
	plt.legend()
	plt.show()
