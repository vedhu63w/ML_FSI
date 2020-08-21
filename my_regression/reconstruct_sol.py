import argparse
import numpy as np
import sys

from scipy.io		import FortranFile

from class_main		import make_plots, read_input
from plt_prediction import read_gt, read_pred

sys.path.append('/home/patel.3140/python_my_utils/')
from my_utils		import writable


#this reads the whole grid even the co-ordinates that were not used training
def read_input_full(dir_nm, fl_nm, is_normalize=True):
	f = FortranFile("../ML/%s/%s.q"%(dir_nm, fl_nm))
	f.read_ints()
	f.read_ints()
	X = f.read_reals()
	#first 3 dimension are shape of grid. last is the number of modes
	#grids are of shape 286,1,301
	#for plate_1 and plate_2 there are 10 modes while plate_0 has grid for
	# 2 variables - deflection and pressure
	#fortran format - first index changing fastest
	if (fl_nm=="plate_1" or fl_nm=="plate_2"):
		X = X.reshape(286, 301, 10, order='F')
		X = X.reshape(1, 286, 301, 10)
	elif (fl_nm=="plate_0"):
		X = X.reshape(286, 301, 2, order='F')
		X = X.reshape(1, 286, 301, 2)

	X = np.moveaxis(X, 3, 0)
	#normalize
	if (is_normalize):
		for i in range(X.shape[0]):
			X[i] = (X[i]-X[i].mean())/X[i].std()

	return X


def get_temporal_val(freq, amp, phase, t):
	return amp*np.sin(2*np.pi*freq*t+phase)


def main(opts):
	dir_nm = opts.dir_nm
	X_avg_sol = read_input_full(opts.dir_nm, "plate_0", is_normalize=False)
	if (opts.fl_nm=="plate_1"):
		X_avg_sol = X_avg_sol[0][0]
	elif (opts.fl_nm=="plate_2"):
		X_avg_sol = X_avg_sol[1][0]
	X_rom_modes = read_input_full(opts.dir_nm, opts.fl_nm, is_normalize=False)
	
	#model predicted data
	pred_freq_test = read_pred("freq_pred_test_%s_%s"%(dir_nm,opts.fl_nm))
	pred_amp_test = read_pred("ampt_pred_test_%s_%s"%(dir_nm,opts.fl_nm))
	pred_phase_test = read_pred("phase_pred_test_%s_%s"%(dir_nm,opts.fl_nm))

	#gt data
	gt_amp_freq_test = read_gt(dir_nm, opts.fl_nm)
	gt_amp_test = gt_amp_freq_test[:,0]
	gt_freq_test = gt_amp_freq_test[:,1]
	gt_phase_test = gt_amp_freq_test[:,2]
	

	writable(writable("plots", "ReconstructedSolution"), "temp")
	for t in range(1000):
		X_sol = X_avg_sol
		for i in range(10):
			temporal_val = get_temporal_val(pred_freq_test[i], pred_amp_test[i], \
													pred_phase_test[i], t)
			X_sol += X_rom_modes[i][0]*temporal_val
	
		make_plots(X_sol, writable("ReconstructedSolution", \
				("sol_%s_%s_%d"%(opts.dir_nm, opts.fl_nm,t)).replace(".","_")))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir_nm', help='directory name', required=True)
	parser.add_argument('--fl_nm', help='file name', required=True)
	
	opts = parser.parse_args()
	main(opts)
