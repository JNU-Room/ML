# Lab 12 Character Sequence RNN
from lib.rnn_core2 import RNNCore2


class XXX (RNNCore2):

    def init_network(self):
        self.set_placeholder(self.sequence_length) #15

        hypothesis = self.rnn_lstm_cell(self.X, self.num_classes, self.hidden_size, self.batch_size)

        self.set_hypothesis(hypothesis)
        self.set_cost_function(self.batch_size, self.sequence_length)
        self.set_optimizer(0.1)


gildong = XXX()
ms = " If you want you"

xd, yd = gildong.get_data(ms)
print(xd)
print(yd)

gildong.learn(xd, yd, 400, 20) #3000
gildong.predict(xd)
gildong.show_error()



'''
0 loss: 2.29895 Prediction: nnuffuunnuuuyuy
1 loss: 2.29675 Prediction: nnuffuunnuuuyuy
2 loss: 2.29459 Prediction: nnuffuunnuuuyuy
3 loss: 2.29247 Prediction: nnuffuunnuuuyuy

...

1413 loss: 1.3745 Prediction: if you want you
1414 loss: 1.3743 Prediction: if you want you
1415 loss: 1.3741 Prediction: if you want you
1416 loss: 1.3739 Prediction: if you want you
1417 loss: 1.3737 Prediction: if you want you
1418 loss: 1.37351 Prediction: if you want you
1419 loss: 1.37331 Prediction: if you want you
'''
