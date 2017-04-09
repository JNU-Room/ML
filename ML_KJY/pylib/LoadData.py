import numpy as np


#파일로 데이터를 읽어오는 클래스
#데이터 형식
# x1 x2 x3 y1
# ?  ?  ?  ?
class LoadData:
	x_data=None
	y_data=None
	def __init__(self,filename,x_length):
		xy = np.loadtxt(filename, unpack=True, dtype='float32')
		self.x_data = xy[0:x_length]
		self.y_data = xy[-1]
