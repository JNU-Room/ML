from pylib.LenearRegression import LenearRegression
from pylib.LoadData import LoadData
from pylib.LogisticClassification import LogisticClassification
from pylib.Softmax import Softmax

pack = LoadData('train.txt')

x_data = [[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.]]
y_data = [[1.,0.,0.,0.,0.],[0.,1.,1.,0.,0.],[0.,0.,0.,1.,1.]]
test = Softmax(x_data,y_data)

test.training(0.7,2001,True)
# test.show_cost_graph()
# test.show_singlevariable_graph()
test.predict([[0.,1.,2.,3.,4.,5.],[0.,1.,2.,3.,4.,5.]])