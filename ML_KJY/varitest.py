from pylib.New_LenearRegression3 import New_LenearRegression
from pylib.LenearRegression import LenearRegression


x_data = [[4.], [5.], [6.], [7.], [8.], [9.], [10.], [11.]]
x_dic =[]
y_data = [[4.], [5.5], [7.5], [4.5], [7], [9], [9], [9.5]]

for i in range(len(x_data[0])):
    x_dic.append([])
    for j in range(len(x_data)):
        if x_data[j][i] not in x_dic[i] :
            x_dic[i].append(x_data[j][i])
    x_dic[i].sort()

lenear = New_LenearRegression(x_data, y_data, x_dic)
# lenear = L
lenear.set_cost(x_data, 1)
# lenear.load_weight()
lenear.training(0.001, 5000, True)
lenear.show_singlevariable_graph()
lenear.predict([[4.], [5.], [6.], [7.], [8.], [9.], [10.], [11.], [4.6] ,[8.1] ,[7.6] , [10.3]])