import argparse
from matplotlib		import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import spearmanr

from class_main	import read_input


def read_gt(dir_nm, fl_nm):
	gt_amp_freq = np.loadtxt("../freq_ampt/my_Results/%s/%s_at.txt"%(dir_nm, \
												fl_nm), delimiter=',')
	gt_amp_freq = gt_amp_freq[:10,:]
	return gt_amp_freq


def read_pred(fl_nm):
	pred_freq = []
	with open("Results/%s"%fl_nm) as fl_in:
		line = fl_in.readline()
		while (line):
			pred_freq.append(float(line.strip("\n")))
			line = fl_in.readline()
	pred_freq = np.array(pred_freq)
	return pred_freq
	

def do_plt_cnn_pred(gt_freq, gt_amp, pred_amp, pred_freq, fl_nm):
	x = np.array(list(range(1000)))
	
	for i in range(gt_freq.shape[0]):
		data_gt = gt_amp[i]*np.sin(2*np.pi*gt_freq[i]*x)
		data_pred = gt_amp[i]*np.sin(2*np.pi*pred_freq[i]*x)
		#print ("i:%d GT: %.3f pred: %.3f\n"%(i+1, gt_freq[i], pred_freq[i]))
		fig, ax = plt.subplots()
		plt.plot(x, data_gt, label="Sine fit", color='b', linewidth=5.0)
		plt.plot(x, data_pred, label="CNN prediction", color='r', linewidth=5.0)
		plt.gcf().set_size_inches(16, 8)
		plt.xlabel("Timestamp", fontsize=30)
		plt.ylabel("Magnitude", fontsize=30)
		ax.tick_params(axis='both', which='major', labelsize=25)
		plt.legend(loc="upper right", fontsize=30)
		plt.savefig("Results/%s_%d"%(fl_nm, i+1), bbox_inches='tight')
		plt.close()


def do_plt_cnn_pred_phase(gt_freq, gt_amp, gt_phase, pred_amp, pred_freq, pred_phase, fl_nm):
	x = np.array(list(range(1000)))
	
	for i in range(gt_freq.shape[0]):
		data_gt = gt_amp[i]*np.sin(2*np.pi*gt_freq[i]*x + gt_phase[i])
		data_pred = gt_amp[i]*np.sin(2*np.pi*pred_freq[i]*x + pred_phase[i])
		#print ("i:%d GT: %.3f pred: %.3f\n"%(i+1, gt_freq[i], pred_freq[i]))
		fig, ax = plt.subplots()
		plt.plot(x, data_gt, label="Sine fit")
		plt.plot(x, data_pred, label="CNN prediction")	
		plt.gcf().set_size_inches(16, 8)
		plt.xlabel("Timestamp", fontsize=30)
		plt.ylabel("Magnitude", fontsize=30)
		ax.tick_params(axis='both', which='major', labelsize=25)
		plt.legend(loc="upper right", fontsize=30)
		plt.savefig("Results/%s_%d_phase"%(fl_nm, i+1), bbox_inches='tight')
		plt.close()


def do_plt_phase_phi_pred(gt_phase, pred_phase, fl_nm):
	#Adopted from :https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib
	def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
		def gcd(a, b):
			while b:
				a, b = b, a%b
			return a
		def _multiple_formatter(x, pos):
			den = denominator
			num = np.int(np.rint(den*x/number))
			com = gcd(num,den)
			(num,den) = (int(num/com),int(den/com))
			if den==1:
				if num==0:
					return r'$0$'
				if num==1:
					return r'$%s$'%latex
				elif num==-1:
					return r'$-%s$'%latex
				else:
					return r'$%s%s$'%(num,latex)
			else:
				if num==1:
					return r'$\frac{%s}{%s}$'%(latex,den)
				elif num==-1:
					return r'$\frac{-%s}{%s}$'%(latex,den)
				else:
					return r'$\frac{%s%s}{%s}$'%(num,latex,den)
		return _multiple_formatter

	class Multiple:
		def __init__(self, denominator=2, number=np.pi, latex='\pi'):
			self.denominator = denominator
			self.number = number
			self.latex = latex
		def locator(self):
			return plt.MultipleLocator(self.number / self.denominator)
		def formatter(self):
			return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))

	#for i in range(gt_phase.shape[0]):
	#	print("gt %f pred %f"%(gt_phase[i], pred_phase[i]))
	fig, ax = plt.subplots()
	x = np.array(range(gt_phase.shape[0]))
	my_xticks = ['%d'%(i+1) for i in range(gt_phase.shape[0])]
	plt.xticks(x, my_xticks)

	fig.set_size_inches(16, 8)
	ax.bar(x+0.00, gt_phase, label="Sine Fit Phase", width=0.33)
	ax.bar(x+0.33, pred_phase, label="CNN Prediction Phase", width=0.33)
	#plt.scatter(x, gt_phase, label="Sine Fit Phase", marker='o',s=gt_phase.shape[0]**3)
	#plt.scatter(x, pred_phase, label="CNN Prediction Phase", marker='o',s=gt_phase.shape[0]**3)
	plt.ylim(-np.pi, np.pi)

	plt.xlabel("Modes", fontsize=30)
	plt.ylabel("Phase", fontsize=30)
	ax.tick_params(axis='both', which='major', labelsize=25)
	
	ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
	ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

	plt.axhline(0, color='black')

	plt.legend(fontsize=25)
	
	#plt.grid()
	plt.tight_layout()
	plt.savefig("Results/%s_phi_all"%(fl_nm), bbox_inches='tight')
	plt.close()

'''
#rewrote this in plt_relative_phase.py
def do_phase_plt(gt_freq, pred_freq, mode_1, mode_2):
	x = np.array(list(range(1000)))
	data_gt_1 = np.sin(2*np.pi*gt_freq[mode_1+1]*x)
	data_gt_2 = np.sin(2*np.pi*gt_freq[mode_2+1]*x)
	plt.plot(data_gt_1, data_gt_2, label="ground truth phase")
	plt.legend()
	plt.savefig("Results/phase_gt_%d_%d"%(mode_1,mode_2))
	plt.close()
	
	data_pred_1 = np.sin(2*np.pi*pred_freq[mode_1+1]*x)
	data_pred_2 = np.sin(2*np.pi*pred_freq[mode_2+1]*x)
	plt.plot(data_pred_1, data_pred_2, label="predicated phase")
	plt.legend()
	plt.savefig("Results/phase_pred_%d_%d"%(mode_1,mode_2))
'''


def main(opts):
	lst_train_dir_nm = opts.train_dir_nm.split(",")
	test_dir_nm = opts.test_dir_nm

	#model predicted data
	pred_freq_train = read_pred("freq_pred_train_%s_%s"%(\
								"_".join(lst_train_dir_nm),opts.fl_nm))
	pred_freq_test = read_pred("freq_pred_test_%s_%s"%(test_dir_nm,opts.fl_nm))
	pred_amp_train = read_pred("ampt_pred_train_%s_%s"%(\
								"_".join(lst_train_dir_nm),opts.fl_nm))
	pred_amp_test = read_pred("ampt_pred_test_%s_%s"%(test_dir_nm,opts.fl_nm))
	pred_phase_train = read_pred("phase_pred_train_%s_%s"%(\
								"_".join(lst_train_dir_nm),opts.fl_nm))
	pred_phase_test = read_pred("phase_pred_test_%s_%s"%(test_dir_nm,opts.fl_nm))

	#train gt data
	gt_amp_freq_1_train = read_gt(lst_train_dir_nm[0], opts.fl_nm)
	gt_amp_freq_2_train = read_gt(lst_train_dir_nm[1], opts.fl_nm)

	gt_amp_freq_train = np.concatenate((gt_amp_freq_1_train, gt_amp_freq_2_train), axis=0)
	gt_amp_train = gt_amp_freq_train[:,0]
	gt_freq_train = gt_amp_freq_train[:,1]
	gt_phase_train = gt_amp_freq_train[:,2]

	#test gt data
	gt_amp_freq_test = read_gt(test_dir_nm, opts.fl_nm)
	gt_amp_test = gt_amp_freq_test[:,0]
	gt_freq_test = gt_amp_freq_test[:,1]
	gt_phase_test = gt_amp_freq_test[:,2]

	#spatial_amp_test = read_input(test_dir_nm, 10, opts.fl_nm, False)
	#import pdb
	#pdb.set_trace()
	
	do_plt_cnn_pred(gt_freq_train, gt_amp_train, pred_amp_train, \
									pred_freq_train, "train_%s_%s"\
									%("_".join(lst_train_dir_nm).replace(".","_"),
														opts.fl_nm))
	do_plt_cnn_pred(gt_freq_test, gt_amp_test, pred_amp_test, pred_freq_test, \
								"test_%s_%s"%(test_dir_nm.replace(".","_"),\
														opts.fl_nm))

	do_plt_cnn_pred_phase(gt_freq_train, gt_amp_train, gt_phase_train,\
									 pred_amp_train, pred_freq_train, \
									pred_phase_train, "train_%s_%s"%\
									("_".join(lst_train_dir_nm).replace(".","_"),\
														opts.fl_nm))
	do_plt_cnn_pred_phase(gt_freq_test, gt_amp_test, gt_phase_test, pred_amp_test,\
									pred_freq_test, pred_phase_test, \
									"test_%s_%s"%(test_dir_nm.replace(".","_"),\
														opts.fl_nm))

	do_plt_phase_phi_pred(gt_phase_train, pred_phase_train, "train_%s_%s"%\
								("_".join(lst_train_dir_nm).replace(".","_"),\
														opts.fl_nm))
	do_plt_phase_phi_pred(gt_phase_test, pred_phase_test, "test_%s_%s"%\
											(test_dir_nm.replace(".","_"),\
														opts.fl_nm))
	
	coef_freq, p_freq = spearmanr(pred_freq_test, gt_freq_test)
	print ("Frequency spearman rank correlation %f and p value %f\n"%(coef_freq, p_freq))
	coef_phase, p_phase = spearmanr(pred_phase_test, gt_phase_test)
	print ("Phase spearman rank correlation %f and p value %f\n"%(coef_phase, p_phase))
	#do_phase_plt(gt_freq_test, pred_freq_test, 1, 2)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_dir_nm', help='train directory name', required=True)
	parser.add_argument('--test_dir_nm', help='test directory name', required=True)
	parser.add_argument('--fl_nm', help='file name', required=True)
	opts = parser.parse_args()
	main(opts)
