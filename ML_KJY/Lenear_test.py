from pylib.LenearRegression import LenearRegression
from pylib.LoadData import LoadData
from pylib.LogisticClassification import LogisticClassification
from pylib.Softmax import Softmax
import numpy

#pack = LoadData('train.txt')
x_data = [[0.,1.,2.,3.,4.],[1.,2.,3.,4.,5.],[3.,4.,5.,6.,7.],[11.,12.,13.,14.,15.],[1,2,3,4,5]]
y_data = [[0.],[2.],[1.],[6.],[4.]]
test = LenearRegression(x_data,y_data)
# l1 = test.create_layer(x_data,5, 4)
# l2 = test.create_layer(l1, 4, 3)
# l3 = test.create_layer(l2, 3, 4)
test.set_cost(x_data ,5)
test.training(0.001,2001,True)
test.show_cost_graph()
print('-----------------')
test.predict([[0.,1.,2.,3.,4.],[1.,2.,3.,4.,5.],[3.,4.,5.,6.,7.],[11.,2.,5.,7.,15.]])

