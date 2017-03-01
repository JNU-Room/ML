from pylib.LenearRegression import LenearRegession
from pylib.LoadData import LoadData

pack = LoadData('train.txt')

print(pack.x_data,pack.y_data)

test = LenearRegession(pack.x_data,pack.y_data)

test.training(0.0001,2001)
#test.show_cost_graph()
#test.show_singlevariable_graph()
test.predict([[1,2,3,4],[1,2,3,4],[1,2,3,4]])