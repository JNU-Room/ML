from pylib.New_LenearRegression import New_LenearRegression
from pylib.LenearRegression import LenearRegression
from pylib import Transposition as tp

from pylib.LoadData import LoadData

#x_dic = [[20,21,22,23,24,25,26,27,30],[0,1,2,3,4,5],[1,2],[21,22,23,24,25,25.5,26,27,28,29,30,34],[0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11,14]]
x_dic = []

xxx = LoadData('sleepdata.dat', 5)

x_data = xxx.x_data
x_data = tp.transpose(x_data)
y_temp = xxx.y_data
for i in range(len(x_data[0])):
    x_dic.append([])
    for j in range(len(x_data)):
        if x_data[j][i] not in x_dic[i] :
            x_dic[i].append(x_data[j][i])
    x_dic[i].sort()

y_data = []
for i in y_temp:
    y_data.append([i])

print(y_data)
# lenear = New_LenearRegression(x_data, y_data, x_dic)
lenear = LenearRegression(x_data, y_data)

lenear.set_cost(x_data, 5)
# lenear.load_weight()
lenear.training(0.0005, 2000000, True)
lenear.save_weight()
# lenear.show_cost_graph()

prd = [[20,1,1,24,7],[21,1,1,24,7],[22,1,1,24,7],[23,1,1,24,7],[24,1,1,24,7]]  # 5

prd2 = [[22,1,1,22,7],[22,1,1,23,7],[22,1,1,24,7],[22,1,1,25,7],[22,1,1,26,7],[22,1,1,27,7],[22,1,1,28,7]]

prd3 = [[22, 1, 1, 24, 4],[22,1,1,24,5],[22,1,1,24,6],[22,1,1,24,7],[22,1,1,24,8],[22,1,1,24,9],[22,1,1,24,10],[22,1,1,24,11],[22,1,1,24,12],[22,1,1,24,13]]

prd4 = [[22, 1, 1, 24, 7],[22,1,2,24,7]]
lenear.predict(prd)
lenear.predict(prd2)
lenear.predict(prd3)
lenear.predict(prd4)
'''[[  8.30378437]
 [ 10.26174068]
 [  9.06564999]
 [ 10.15423584]
 [  9.81656361]]'''