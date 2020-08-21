from scipy.io import FortranFile
import numpy as np
import glob


'''
f=FortranFile("./h0.008/plate_2.x", "r")
print(f.read_ints())
print(f.read_ints())

t=f.read_reals()
print (t.shape)
print (type(t[0]))
print (t.reshape(286,1,301,3).shape)
print(t[0:10])

f=FortranFile("./h0.008/plate_2.q", "r")

print(f.read_ints())
print(f.read_ints())
'''


def read_input(dir_nm, fl_nm):
	lst_fl_x = glob.glob('../ML/%s/COMBplate/*.x'%(dir_nm))
	lst_fl_x = lst_fl_x[:100]
	X = np.zeros((len(lst_fl_x), 3, 286, 301))
	for i, fl_x in enumerate(lst_fl_x):
		f=np.fromfile(fl_x, dtype='<i4')
		x = f[4:].view(dtype='<f4')
		#print (x.shape)
		assert(x.shape[0]%(286*301) == 0)
		x = x.reshape([3,286,301])
		X[i] = x
	return X
