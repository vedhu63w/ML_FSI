from __future__ import division

'''This fits sine wave to temporal modes. The methodology is adopted from '''
'''https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy'''


import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
from scipy import optimize
from sklearn.metrics import mean_squared_error

import os
import sys

sys.path.append('/home/patel.3140/python_my_utils/')
from my_utils import writable


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega"'''
    ''', "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    # excluding the zero frequency "peak", which is related to offset
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    val = fitfunc(tt)
    mse_val = mean_squared_error(val, yy)
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, \
			"period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), \
							"rawres": (guess,popt,pcov), "mse": mse_val}


def main(opts):
	dir_nm = opts.dir_nm
	fl_nm = opts.fl_nm
	np_data = np.loadtxt('../Madhav/%s/%s.txt'%(dir_nm, fl_nm))
	lst_mse = []
	with open(writable(writable("my_Results",dir_nm), "%s.txt"%fl_nm), "w") as fl_out:
		for i in range(2, 12):
			x,y=np_data[:,0], np_data[:,i]
			res = fit_sin(x, y)
			if (res["amp"]<0):
				res["amp"] = -res["amp"]
				res["phase"] = res["phase"] + np.pi
			#TODO: assumes initial phase is between -pi to pi
			if (res["phase"] > np.pi):
				res["phase"] = res["phase"] - 2*np.pi
			elif (res["phase"] < -np.pi):
				res["phase"] = res["phase"] + 2*np.pi
			print( "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )
	
			lst_mse.append(res["mse"])
			fig, ax = plt.subplots()
			plt.scatter(x, y, label="Temporal Mode points", color='green', linewidth=0.3)
			plt.plot(x, res["fitfunc"](x), label="Sine fit", color='blue')
			plt.gcf().set_size_inches(16, 8)
			plt.xlabel("Timestamp", fontsize=30)
			plt.ylabel("Magnitude", fontsize=30)
			ax.tick_params(axis='both', which='major', labelsize=25)
			plt.legend(loc='upper right', fontsize=30)
			plt.savefig(writable(writable(writable("my_Results",dir_nm), fl_nm), "%d"%(i-2)), \
															bbox_inches='tight')
			plt.close()
			fl_out.write("%f, %f, %f\n"%(abs(res["amp"]), res["freq"], res["phase"]))
	print ("Mean mse: %f"%(sum(lst_mse)/len(lst_mse)))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir_nm', help='directory name', required=True)
	parser.add_argument('--fl_nm', help='file name', required=True)
	opts = parser.parse_args()
	main(opts)
