# Lab 12 RNN
from lib.sentence_multi_rnn import SentenceMultiRNN


class XXX (SentenceMultiRNN):

    def init_network(self):
        self.set_placeholder(self.length_of_sequence) # 10

        hypo = self.create_multi_rnn_layer()

        self.set_hypothesis(hypo)
        self.set_cost_function()
        self.set_optimizer(0.1)


sent = ("if you want to build a ship, don't drum up people together to "
        "collect wood and don't assign them tasks and work, but rather "
        "teach them to long for the endless immensity of the sea.")
sent2 = (" 내가 만일 하늘이라면 그대 얼굴에 물들고 싶어.. 붉게 물든 저녁 저 노을처럼.. 나 그대 뺨에 물들고 싶어.." 
         " 내가 만일 시인이라면 그대 위해 노래하겠어.. 엄마품에 안긴 어린 아이처럼.. 나 행복하게 노래하고 싶어.."
         " 세상에 그 무엇이라도 그대 위해 되고 싶어.. 오늘처럼 우리 함께 있음이 내겐 얼마나 큰 기쁨인지..")

gildong = XXX()

gildong.set_parameters(sent2, 10)
dataX_, dataY_ = gildong.sentence_to_data(sent2)
gildong.learn(dataX_, dataY_, 100, 10)
print(gildong.hidden_size, gildong.number_of_class, gildong.batch_size, gildong.length_of_sequence, )
gildong.predict(dataX_)
#gildong.show_error()

