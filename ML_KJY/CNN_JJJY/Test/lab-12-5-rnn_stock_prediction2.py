from lib.common_db import CommonDB
from lib.stock_rnn import StockRNN


class MyDB(CommonDB):
    def preprocesisng(self):
        self.reverse()
        self.min_max_scalr()
        #self.normalize()


class MyRNN(StockRNN):

    def init_network(self):
        self.set_parameter(5, 7, 1) #input_size, seq_length, output_size
        self.set_placeholder(self.seq_length, self.input_dim)

        hypo = self.create_simple_rnn_layer(self.output_dim)

        self.set_hypothesis(hypo)
        self.set_cost_function()

        self.set_optimizer(0.01)


gildong = MyRNN()

db = MyDB()
db.load('data-02-stock_daily.csv', 7) # seq_length

gildong.learn(db.trainX, db.trainY, 100, 10)
gildong.show_error()
gildong.predict(db.testX, db.testY)


'''
in case that total_loop = 100,
96 0.692995
97 0.692761
98 0.692537
99 0.692321
RMSE 0.0372195926237
'''
