from pylib.LenearRegression import LenearRegression
from pylib.LoadData import LoadData
from pylib.LogisticClassification import LogisticClassification

pack = LoadData('train.txt')

x_data = [0.,1.,2.,3.,4.,5.,6.,7.,8.]
y_data = [0.,0.,0.,0.,0.,0.,1.,1.,1.]
test = LogisticClassification(x_data,y_data)

test.training(0.3,2001,True)
test.show_cost_graph()
test.show_singlevariable_graph()
test.predict([0.,1.,2.,3.,4.,5.])