import numpy as np

class LoadData:
	x_data=None
	y_data=None
	def __init__(self,filename):
		xy = np.loadtxt(filename, unpack=True, dtype='float32')
		self.x_data = xy[0:-1]
		self.y_data = xy[-1]
