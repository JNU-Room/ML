# Lab 12 RNN
from lib.rnn_core3 import RNNCore3
from lib.tensor_board_util import TensorBoardUtil
from lib.string_db3 import StringDB3


class XXX (RNNCore3):
    board = TensorBoardUtil()

    def init_network(self):
        self.set_placeholder(seq_len=self.length_of_sequence, hidden_size=self.hidden_size)

        logits = self.rnn_lstm_cell(self.X, num_classes=self.input_size, hidden_size=self.hidden_size, batch_size=1)

        self.set_hypothesis(logits)
        self.set_cost_function(batch_size=1, seq_len=self.length_of_sequence)
        self.set_optimizer(0.1)

        #self.tbutil.scalar('Cost', self.cost_function)
        #self.tbutil.merge()

    def create_writer(self):
        #self.tbutil.create_writer(self.sess, './tb/rnn01')
        pass

    def do_summary(self, feed_dict):
        #self.tbutil.do_summary(self.sess, feed_dict)
        pass


gildong = XXX()

db = StringDB3()
db.load(' Hello,World!')

gildong.set_parameters(db.unique_char_num, db.sequence_num) # '유일한 문자수, 전체길이-1
gildong.show_parameters()

gildong.learn(db.x_one_hot, db.y_one_hot, 500, 10)

index = gildong.predict(db.sentence_to_index_list(' Hello,World')) # 'hello,world!'
print(db.index_list_to_sentence(index))

index = gildong.predict(db.sentence_to_index_list(' Hello,Worll')) # 'hello,world!'
print(db.index_list_to_sentence(index))

gildong.show_error()

