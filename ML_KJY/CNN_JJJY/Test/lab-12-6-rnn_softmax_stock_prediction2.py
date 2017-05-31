from lib.common_db import CommonDB
from lib.stock_rnn import StockRNN


class MyStockDB(CommonDB):
    def preprocesisng(self):
        self.reverse()
        self.normalize()
        #self.min_max_scalr()


class MyRNN(StockRNN):
    def init_network(self):
        self.set_parameter(5, 7, 30) #input, seq_length, output
        self.set_placeholder(self.seq_length, self.input_dim)

        hypo = self.create_multi_rnn_softmax_layer()

        self.set_hypothesis(hypo)
        self.set_cost_function()
        self.set_optimizer(0.001)


gildong = MyRNN()

db = MyStockDB()
db.load('data-02-stock_daily.csv', 7) # gildong.seq_length

gildong.learn(db.trainX, db.trainY, 500, 10)
gildong.predict(db.testX, db.testY)
#gildong.show_error()


'''
96 1.27205
97 1.27146
98 1.27088
99 1.27032
RMSE 0.0884736615436
'''


