import tensorflow as tf
from New_LenearRegression import New_LenearRegression


sess = tf.Session()

x_data_dic = [[1,2,3,4,5]]

#
# w1 = tf.Variable(tf.random_uniform([5]))
# w2 = tf.Variable(tf.random_uniform([3]))
# w3 = tf.Variable(tf.random_uniform([3]))

x_data = [[2],[3]]
y_data = [[2],[3],[4],[5]]

#
# for i in range(len(x_data)):
#     for j in range(len(x_data[0])):
#         x = x_data_dic[j].index(x_data[i][j])
#         if j == 0 :
#             x_data[i][j] = w1[x]
#         elif j == 1 :
#             x_data[i][j] = w2[x]
#         elif j == 2 :
#             x_data[i][j] = w3[x]

test = New_LenearRegression(x_data, y_data,x_data_dic)
test.set_cost(x_data ,3)
test.training(0.0005,5001,True)
test.show_cost_graph()
print('-----------------')
test.predict([[2,4,5],[1,2,5],[3,4,5],[5,3,6]])
