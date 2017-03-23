import tensorflow as tf
import numpy as np

# not moldule
sess = tf.Session()

char_rdic = ['my', 'name', 'is', 'Kim', 'Jae', 'Yun', '!','gg', 'Han', 'Sung'] # id -> char
char_dic = {w : i for i, w in enumerate(char_rdic)} # char -> id
print (char_dic)


ground_temp = 'my name is Kim Jae Yun ! gg gg gg gg ! ! !'
ground_temp2 = 'my name is Han Sung ! gg gg gg gg ! ! !'
ground_temp = ground_temp.split(' ')
ground_temp2 = ground_temp2.split(' ')
ground_truth = [char_dic[c] for c in ground_temp[:11]]
ground_truth2 = [char_dic[c] for c in ground_temp2[:11]]
print (ground_truth)

x_data = np.array([[1,0,0,0,0,0,0,0,0,0], # my
                   [0,1,0,0,0,0,0,0,0, 0], # name
                   [0,0,1,0,0,0,0,0, 0, 0], # is
                   [0,0,0,1,0,0,0,0, 0, 0], # Kim,
                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # Jae
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # Yun
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # !
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # gg
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # Han
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],dtype= np.float32) # Sung


# Configuration
rnn_size = len(char_dic) # 7
batch_size = 1
output_size = 7



# RNN Model
rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units = rnn_size,
                                       input_size = None, # deprecated at tensorflow 0.9
                                       #activation = tanh,
                                       )

print(rnn_cell)

initial_state = rnn_cell.zero_state(batch_size, tf.float32)
print(initial_state)



initial_state_1 = tf.zeros([batch_size, rnn_cell.state_size]) #  위 코드와 같은 결과
print(initial_state_1)


print (x_data)
x_split = tf.split(x_data, rnn_size, 0) # 가로축으로 ?개로 split
#
# sess.run(x_split)
# sess.run(x_split[0])
print(x_split)

#outputs, state = tf.nn.rnn(cell = rnn_cell, inputs = x_split, initial_state = initial_state)   #구버전
outputs, state = tf.contrib.rnn.static_rnn(cell = rnn_cell, inputs = x_split, initial_state = initial_state)

print (outputs)
print (state)

logits = tf.reshape(tf.concat(outputs,1), # shape = 1 x 16 버전업으로 파라미터 순서바뀜
                    [-1,rnn_size])        # shape = 4 x 4
logits.get_shape()

targets = tf.reshape(ground_truth[1:], [-1]) # a shape of [-1] flattens into 1-D\
targets2 = tf.reshape(ground_truth2[1:], [-1]) # a shape of [-1] flattens into 1-D
targets.get_shape()
targets2.get_shape()


sess.run(tf.initialize_all_variables())

weights = tf.ones([len(char_dic) * batch_size])
# seq2seq.sequence_loss의 오류검출 코드에서 리스트는 안되고 꼭 텐서로 해야 된다고 합니다.
# 하지만 2,1,1차원 변수를 함수가 요구하는 3,2,2차원 변수로 만들기 위해 이렇게 조정해줍니다.
d3_logics = tf.Variable([logits,logits])
d2_targets = tf.Variable([targets,targets2])
d2_weights = tf.Variable([weights,weights])
#loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights]) <<원랜 이게 됐었다고합니다.
loss = tf.contrib.seq2seq.sequence_loss(d3_logics, d2_targets, d2_weights)
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# Launch the graph in a session
sess.run(tf.initialize_all_variables())
for i in range(100):
    sess.run(train_op)
    result = sess.run(tf.argmax(d3_logics[0], 1))
    result2 = sess.run(tf.argmax(d3_logics[1], 1))
    if i% 20 == 0:
        print(sess.run(cost))
        print(result, [char_rdic[t] for t in result])
        print(result2, [char_rdic[t] for t in result2])
print(sess.run(d2_weights))


def predic(str):
    str = str.split(' ')
    kkk = tf.split(str, rnn_size, 0)
    xxx = []

    for i in kkk:
        xxx.append(char_dic[i])
    print(xxx)
    returnx = sess.run(tf.argmax([xxx], 1))
    print(returnx)
print(sess.run(state))
predic('my name is Kim')


