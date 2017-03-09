from pylib.LenearRegression import LenearRegression
from pylib.LoadData import LoadData
from pylib.LogisticClassification import LogisticClassification
from pylib.Softmax import Softmax
import numpy

pack = LoadData('train.txt')
x_data = [[0.,1.,2.,3.],[1.,2.,3.,4.],[3.,4.,5.,6.],[11.,12.,13.,14.]]
y_data = [[0.,0.,1.,0.],[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]]
test = Softmax(x_data,y_data)

test.training(0.01,2001,True)
# test.show_cost_graph()
# test.show_singlevariable_graph()
print('-----------------')
test.predict([[0.,1.,2.,3.],[1.,2.,3.,4.],[3.,4.,5.,6.],[11.,12.,13.,14.]])

