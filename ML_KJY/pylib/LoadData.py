import numpy as np


#파일로 데이터를 읽어오는 클래스
#데이터 형식
# x1 x2 x3 y1 y2 y3
# ?  ?  ?  ?  ?  ?
class LoadData:
	x_data=None
	y_data=None
	def __init__(self,filename):
		xy = np.loadtxt(filename, unpack=True, dtype='float32')
		self.x_data = xy[0:-1]
		self.y_data = xy[-1]
