from tf_showgraph_module import LenearRegession

x_data =  [[1.,2.,3.,4.,9.],[1.,2.,3.,4.,5.],[4.,5.,6.,7.,8.]]
y_data = [1.,2.,3.,5.,8.]

test = LenearRegession(x_data,y_data)

test.training(0.04,2001)
test.predict([[2,3,4,5],[1,2,3,4],[7,8,6,5]])
test.show_cost_graph()
