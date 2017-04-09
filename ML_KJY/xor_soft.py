from pylib.Softmax import Softmax

softmax = Softmax([[0,0],[0,1],[1,0],[1,1]]
                  ,[[0],[1],[1],[0]])
softmax.set_whc(2)
softmax.training(0.001,5000,True)
softmax.predict([[0,0],[0,1],[1,0],[1,1]])