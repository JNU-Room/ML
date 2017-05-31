from pylib.Softmax import Softmax
import tensorflow as tf

x_data = [[0.,0.],[0.,1.],[1.,0.]]
y_data = [[0.,1.],[1.,0.],[1.,0.]]

softmax = Softmax(x_data,y_data)
l1 = softmax.create_layer(x_data,2,2)
softmax.set_cost(x_data, 2)
softmax.training(0.1,3000,True)
softmax.predict([[0., 0.],[0., 1.],[1., 0.]])
softmax.show_cost_graph()
print(softmax.return_predict_possibility())