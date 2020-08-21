import argparse
import matplotlib.pyplot as plt
import numpy as np
from class_main		import read_input


def plt_amplt_corr(ampt, max_vals, xlbl, ylbl, fl_nm):
	fig, ax = plt.subplots()
	plt.scatter(max_vals, ampt)
	n = range(len(max_vals))
	for i, txt in enumerate(n):
		ax.annotate(txt, (max_vals[i], ampt[i]))
	plt.xlabel(xlbl, fontsize=25)
	plt.ylabel(ylbl, fontsize=25)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.gcf().set_size_inches(16, 8)
	plt.savefig(fl_nm+".png", bbox_inches='tight')
	plt.close()


def main(opts):
	lst_dir_nm = opts.lst_dir_nm.split(',')
	fl_nm = opts.fl_nm
	tot_mode = 10
		
	X = read_input(lst_dir_nm[0], tot_mode, fl_nm, False)
	ampt = np.loadtxt('../freq_ampt/my_Results/%s/%s_at.txt'%(lst_dir_nm[0], \
											fl_nm), delimiter=',')[:tot_mode,0]
	max_vals = np.zeros(tot_mode)
	min_vals = np.zeros(tot_mode)
	for mode in range(tot_mode):
		max_vals[mode] = X[mode,0,:,:].max()
		min_vals[mode] = X[mode,0,:,:].min()
	
	
	plt_amplt_corr(ampt, max_vals, "Maximum value", "Amplitude", \
					"plots/maxval_ampt_%s_%s"%(lst_dir_nm[0],fl_nm))
	plt_amplt_corr(ampt, min_vals, "Minimum value", "Amplitude", \
					"plots/minval_ampt_%s_%s"%(lst_dir_nm[0],fl_nm))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--lst_dir_nm', help='list of directory name', required=True)
	parser.add_argument('--fl_nm', help='file name', required=True)
	opts = parser.parse_args()
	main(opts)
