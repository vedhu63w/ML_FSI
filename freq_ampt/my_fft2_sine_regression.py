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

sys.path.append('../../python_my_utils/')
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
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, \
			"period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), \
											"rawres": (guess,popt,pcov)}


def fit_fft2(tt, yy):
	tt = np.array(tt)
	yy = np.array(yy)
	ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))
	Fyy = abs(np.fft.fft(yy))

	argsort_Fyy = np.flip(np.argsort(Fyy[1:]), axis=0)
	#top 2 sine waves
	guess_freq_1 = abs(ff[argsort_Fyy[0]+1])
	guess_freq_2 = abs(ff[argsort_Fyy[2]+1])
	guess_ampt_1 = 2*Fyy[argsort_Fyy[0]+1]/tt.shape[0]
	guess_ampt_2 = 2*Fyy[argsort_Fyy[2]+1]/tt.shape[0]
	guess_offset = np.mean(yy)
	guess = np.array([guess_ampt_1, guess_ampt_2, 2*np.pi*guess_freq_1, 2*np.pi*guess_freq_2, \
						0, 0, guess_offset])
	def sin2func(t, A1, A2, w1, w2, p1, p2, c): return A1 * np.sin(w1*t + p1) + \
													A2 * np.sin(w2*t + p2)+ c
	popt, pcov = optimize.curve_fit(sin2func, tt, yy, p0=guess, maxfev=10000)
	A1, A2, w1, w2, p1, p2, c = popt
	f1, f2 = w1/(2.*np.pi), w2/(2.*np.pi)
	fitfunc = lambda t: A1 * np.sin(w1*t + p1) + A2 * np.sin(w2*t + p2) + c
	val = fitfunc(tt)
	mse_val = mean_squared_error(val, yy)
	return {"amp": [A1, A2], "omega": [w1, w2], "phase": [p1,p2], "offset": c,\
						"freq": [f1,f2], "fitfunc": fitfunc, "mse":mse_val}


#converts to sine wave with positive amplitude and phase to be between -pi to pi
def cnvt_to_std_sine_form(amp, phase):
	if (amp<0):
		amp = -amp
		phase = phase + np.pi
	while (phase>np.pi): phase = phase - 2*np.pi
	while (phase<-np.pi): phase = phase + 2*np.pi
	assert(amp>0)
	assert(phase<np.pi and phase>-np.pi)
	return (amp, phase)


def main(opts):
	dir_nm = opts.dir_nm
	fl_nm = opts.fl_nm
	np_data = np.loadtxt('../Madhav/%s/%s.txt'%(dir_nm, fl_nm))
	lst_mse = []
	with open(writable(writable("my_Results",dir_nm), "%s_ord2.txt"%fl_nm), "w") as fl_out:
		for i in range(2, 12):
			x,y=np_data[:,0], np_data[:,i]
			# res = fit_sin(x, y)
			# print( "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )
			res = fit_fft2(x, y)
			amp_phase = cnvt_to_std_sine_form(res["amp"][0], res["phase"][0])
			res["amp"][0], res["phase"][0] = amp_phase[0], amp_phase[1]
			amp_phase = cnvt_to_std_sine_form(res["amp"][1], res["phase"][1])
			res["amp"][1], res["phase"][1] = amp_phase[0], amp_phase[1]

			lst_mse.append(res["mse"])
			fig, ax = plt.subplots()
			plt.scatter(x, y, label="Temporal Mode points", color='green', linewidth=0.3)
			plt.plot(x, res["fitfunc"](x), label="sine fit", color='blue')
			plt.gcf().set_size_inches(16,8)
			plt.xlabel("Timestemp", fontsize=30)
			plt.ylabel("Magnitude", fontsize=30)
			ax.tick_params(axis="both", which="major", labelsize=25)
			plt.legend(loc='upper right', fontsize=30)
			plt.savefig(writable(writable(writable("my_Results",dir_nm), fl_nm), "%d_ord2"%(i-2)),\
																			bbox_inches='tight')
			plt.close()
			# fl_out.write("%f, %f\n"%(abs(res["amp"]), res["freq"]))
			fl_out.write("%f, %f, %f, %f, %f, %f\n"%(res["amp"][0], res["amp"][1], \
						res["freq"][0], res["freq"][1], res["phase"][0], res["phase"][1]))
	print ("Mean mse: %f"%(sum(lst_mse)/len(lst_mse)))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir_nm', help='directory name', required=True)
	parser.add_argument('--fl_nm', help='file name', required=True)
	opts = parser.parse_args()
	main(opts)
