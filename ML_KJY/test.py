from pylib.LenearRegression import LenearRegession
from pylib.LoadData import LoadData
from pylib.LogisticClassification import LogisticClassification

pack = LoadData('train.txt')

x_data = [[0.,1.,2.,3.,4.,5.],[4.,5.,6.,7.,8.,9.]]
y_data = [0.,1.,2.,3.,4.,5.]
test = LogisticClassification(x_data,y_data)

test.training(0.0001,2001,True)
# test.show_cost_graph()
# test.show_singlevariable_graph()
test.predict([[0.,1.,2.,3.,4.,5],[4.,5.,6.,7.,8.,9.]])