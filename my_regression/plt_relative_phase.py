import argparse
import matplotlib.pyplot as plt
import numpy as np

from plt_prediction import read_pred, read_gt


def plt_relative_phase(gt_freq, gt_amp, gt_phase, pred_freq, pred_amp, \
											pred_phase, modes, out_fl_nm):
	fig, ax = plt.subplots()
	mode_1, mode_2 = int(modes[0]), int(modes[1])
	#x = np.array(list(range(1000)))
	x = np.linspace(0, 100*np.pi, num=1000)
	data_gt_1 = gt_amp[mode_1-1]*np.sin((2*np.pi*gt_freq[mode_1-1])*x)
	data_gt_2 = gt_amp[mode_2-1]*np.sin((2*np.pi*gt_freq[mode_2-1])*x)
	#data_gt_1 = np.sin((2*np.pi*0.1)*x)
	#data_gt_2 = np.sin((2*np.pi*0.2)*x)
	
	data_pred_1 = gt_amp[mode_1-1]*np.sin(2*np.pi*pred_freq[mode_1-1]*x)
	data_pred_2 = gt_amp[mode_2-1]*np.sin(2*np.pi*pred_freq[mode_2-1]*x)

	plt.plot(data_gt_1, data_gt_2, label="Sine fit")
	plt.plot(data_pred_1, data_pred_2, label="CNN prediction")
	ax.tick_params(axis="both", which='major', labelsize=25)
	plt.gcf().set_size_inches(16, 8)
	plt.xlabel("Mode %d"%(mode_1), fontsize=30)
	plt.ylabel("Mode %d"%(mode_2), fontsize=30)
	plt.legend(fontsize=30)
	plt.savefig("Results/%s_%d_%d"%(out_fl_nm, mode_1, mode_2), \
											bbox_inches='tight')


def main(opts):
	lst_train_dir_nm = opts.train_dir_nm.split(",")
	test_dir_nm = opts.test_dir_nm

	#model predicted data
	pred_freq_test = read_pred("freq_pred_test_%s_%s"%(test_dir_nm,opts.fl_nm))
	pred_amp_test = read_pred("ampt_pred_test_%s_%s"%(test_dir_nm,opts.fl_nm))
	pred_phase_test = read_pred("phase_pred_test_%s_%s"%(test_dir_nm,opts.fl_nm))

	#test gt data
	gt_amp_freq_test = read_gt(test_dir_nm, opts.fl_nm)
	gt_amp_test = gt_amp_freq_test[:,0]
	gt_freq_test = gt_amp_freq_test[:,1]
	gt_phase_test = gt_amp_freq_test[:,2]

	plt_relative_phase(gt_freq_test, gt_amp_test, gt_phase_test, pred_freq_test, \
						pred_amp_test, pred_phase_test, opts.modes.split(","), \
						"test_relv_phase_%s"%(test_dir_nm.replace(".", "_")))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_dir_nm", help='train directory name', \
														required=True)
	parser.add_argument("--test_dir_nm", help='test directory name', \
														required=True)
	parser.add_argument("--fl_nm", help='file name', required=True)
	parser.add_argument("--modes", help='modes for which relative phase is to be computed', \
																required=True)	
	opts = parser.parse_args()
	main(opts)
