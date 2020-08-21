import argparse
import matplotlib.pyplot as plt
import numpy as np


from plt_prediction	import read_pred, read_gt


def plt_survival_plot(vals, fl_out):
	vals = sorted(vals, reverse=True)
	tot = sum(vals)
	y = [float(v)/tot for v in vals]

	fig, ax = plt.subplots()
	plt.plot(range(len(vals)), y, marker='o')
	ax.tick_params(axis="both", which='major', labelsize=25)
	plt.ylabel("% Energy", fontsize=30)
	plt.xlabel("Mode number", fontsize=30)
	plt.yscale('log', basey=10)
	plt.gcf().set_size_inches(16, 8)
	plt.savefig(fl_out+".png")
	plt.close()


def main(opts):
	lst_train_dir_nm = opts.train_dir_nm.split(",")
	test_dir_nm = opts.test_dir_nm
	
	#model predicted data
	pred_amp_test = read_pred("ampt_pred_test_%s"%(test_dir_nm))	
	
	#test gt data
	gt_amp_freq_test = read_gt(test_dir_nm, opts.fl_nm)
	gt_amp_test = gt_amp_freq_test[:,0]
	plt_survival_plot(gt_amp_test, "plots/energy_%s_%s"%(lst_train_dir_nm[0],opts.fl_nm))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_dir_nm", help='train directory name', required=True)
	parser.add_argument("--test_dir_nm", help='test directory name', required=True)
	parser.add_argument("--fl_nm", help='file name', required=True)
	opts = parser.parse_args()
	main(opts)
