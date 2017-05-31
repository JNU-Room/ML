from New_LenearRegression import New_LenearRegression
from LenearRegression import LenearRegression
import Transposition as tp

from LoadData import LoadData

x_dic = []

xxx = LoadData('sleepdata.dat', 5)

x_data = xxx.x_data
x_data = tp.transpose(x_data)
tt =  list(tp.transpose(x_data))
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

linear = New_LenearRegression(x_data, y_data, x_dic)
# linear = LenearRegression(x_data, y_data)

linear.set_cost(x_data, 5)

linear.load_weight()
# linear.training(0.13, 20000, True)
# linear.save_weight()
# lenear.show_cost_graph()

linear.show(tt)

prd = [[20,1,1,24,7],[21,1,1,24,7],[22,1,1,24,7],[23,1,1,24,7],[24,1,1,24,7]]  # 5

prd2 = [[22,1,1,22,7],[22,1,1,23,7],[22,1,1,24,7],[22,1,1,25,7],[22,1,1,26,7],[22,1,1,27,7],[22,1,1,28,7]]

prd3 = [[22, 1, 1, 24, 4],[22,1,1,24,5],[22,1,1,24,6],[22,1,1,24,7],[22,1,1,24,8],[22,1,1,24,9],[22,1,1,24,10],[22,1,1,24,11],[22,1,1,24,12],[22,1,1,24,13]]

prd4 = [[22, 1, 1, 24, 7],[22,1,2,24,7]]

last_prd = [[24,1,1,22,8]]
# linear.predict(prd)
# linear.predict(prd2)
# linear.predict(prd3)
# linear.predict(prd4)
linear.predict(last_prd)

'''[[  8.30378437]
 [ 10.26174068]
 [  9.06564999]
 [ 10.15423584]
 [  9.81656361]]'''