from pylib.LogisticClassification import LogisticClassification
import tensorflow as tf

x_data = [[0.,0.],[0.,1.],[1.,0.],[1.,1.]]
y_data = [[0],[1],[1],[0]]

softmax = LogisticClassification(x_data,y_data)
l1 = softmax.create_layer(x_data,2,3)
l2 = softmax.create_layer(l1,3,4)
l3 = softmax.create_layer(l2,4,5)
softmax.set_cost(l3, 5)
softmax.training(0.08,2000,True)
softmax.show_cost_graph()
softmax.predict([[0.,0.,1.,1.],[0.,1.,0.,1.]])