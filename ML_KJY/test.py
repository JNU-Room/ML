from tf_showgraph_module import LenearRegession

x_data =  [1.,2.,3.,4.,5.]
y_data = [1.,2.,3.,4.,5.]

test = LenearRegession(x_data,y_data)

test.training(0.04,2001)
test.predict([[2,3,4,5],[1,2,3,4]])
test.show_cost_graph()
