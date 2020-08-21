import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plt_prediction		import read_gt, read_pred

sys.path.append('/home/patel.3140/python_my_utils/')
from my_utils			import writable


def main(opts):
	dir_nm = opts.dir_nm
	fl_nm = opts.fl_nm

	#freq amp from dnn
	pred_freq = read_pred("freq_pred_test") 
	pred_amp = read_pred("amp_pred_test") 
	
	#freq amp from fft
	gt_amp_freq = read_gt(dir_nm, fl_nm)
	gt_amp = gt_amp_freq[:,0] 
	gt_freq = gt_amp_freq[:,1] 
	gt_phase = gt_amp_freq[:,2] 

	x = np.array(list(range(1000)))
	
	#actual data from ROM
	np_data = np.loadtxt('../Madhav/%s/%s_at.txt'%(dir_nm, fl_nm))
	data_mode = np_data[:,2:]
	
	lst_mse_fft, lst_mse_dnn = [], []

	with open(writable("evaluate_results", "Results"), "w") as fl_out:
		for i in range(gt_freq.shape[0]):
			data_gt = gt_amp[i]*np.sin(2*np.pi*gt_freq[i]*x+gt_phase[i])
			data_pred = gt_amp[i]*np.sin(2*np.pi*pred_freq[i]*x+gt_phase[i])
			data_mode_i = data_mode[:,i]
		
			plt.plot(x, data_gt, label="FFT")
			plt.plot(x, data_pred, label="DNN")
			plt.plot(x, data_mode_i, label="mode")	
			plt.legend()
			plt.savefig(writable("evaluate_results", "%d"%i))
			plt.close()

			mse_fft = ((data_gt-data_mode_i)**2).mean()
			mse_dnn = ((data_pred-data_mode_i)**2).mean()
			fl_out.write("%.3f, %.3f\n"%(mse_fft,mse_dnn))
			lst_mse_fft.append(mse_fft)
			lst_mse_dnn.append(mse_dnn)
	print ("fft error %.4f"% (sum(lst_mse_fft)/float(len(lst_mse_fft))))
	print ("dnn error %.4f"% (sum(lst_mse_dnn)/float(len(lst_mse_dnn))))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir_nm', help='directory name', required=True)
	parser.add_argument('--fl_nm', help='file name', required=True)
	opts = parser.parse_args()	
	main(opts)
