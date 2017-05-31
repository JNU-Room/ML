# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# http://learningtensorflow.com/index.html
# http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

'''
유닛 = 상태 = 출력 = 히든 수 = 2 : BasicRNNCell 생성시에 줌
dynamic_rnn(cell, x_data(1,1,4)) sequence_length = 1(두번째 1)??
주어지는 데이터는 3차원이며 가장 마지막 것이 입력데이터 차원. 4차원 입력데이터 1개
'''
with tf.variable_scope('cell') as scope:
    # One cell RNN input_dim (4) -> output_dim (2)
    hidden_size = 2
    x_data = np.array([[h]], dtype=np.float32) # x_data = [[[1,0,0,0]]]

    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
    print('size of cell output:', cell.output_size, ', size of cell state:', cell.state_size)

    print(x_data.shape)
    pp.pprint(x_data)

    X = tf.placeholder(tf.float32, [None, 1, 4])
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())

    pp.pprint(sess.run(outputs, feed_dict={X: [[h],[e],[l],[l],[o]]}))

'''
유닛 = 히든 = 출력 = 상태 수 = 2 : BasicRNNCell 생성시에 줌
dynamic_rnn(cell, x_data(1,5,4)) sequence_length = 5(두번째 5)
3차원 입력데이터를 보면 4차원 벡터 5개. 따라서 sequence_length = 5
'''
with tf.variable_scope('sequances') as scope:
    # One cell RNN input_dim (4) -> output_dim (2). sequence: 5
    hidden_size = 2
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
    x_data = np.array([[h, e, l, l, o]], dtype=np.float32) # (1, 5, 4) 5=sequence_len???
    print(x_data.shape)
    pp.pprint(x_data)

    X = tf.placeholder(tf.float32, [None, 5, 4])
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(sess.run(outputs, feed_dict={X: x_data}))

'''
유닛 = 히든 = 출력 = 상태 수 = 2 : BasicRNNCell 생성시 설정
dynamic_rnn(cell, x_data(3,5,4)) 배치:3, 시퀀스:5, 입력:4
입력 데이터(4차원) 5개가 하나의 시퀀스로 묶임(sequence_length = 5)
시퀀스 묶음이 모두 3개 
결국 batch는 시퀀스 묶음 수 = 3 
'''
with tf.variable_scope('batches') as scope:
    # One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
    # 3 batches 'hello', 'eolll', 'lleel'
    x_data = np.array([[h, e, l, l, o], # 배치1
                       [e, o, l, l, l], # 배치2
                       [l, l, e, e, l]], dtype=np.float32) # 배치3
    pp.pprint(x_data)

    X = tf.placeholder(tf.float32, [3,5,4])
    hidden_size = 2
    cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(sess.run(outputs, feed_dict={X: x_data}))

'''
유닛 = 히든 = 출력 = 상태 수 = 2 : BasicRNNCell 생성시 설정
dynamic_rnn(cell, x_data(3,5,4)) 배치:3, 시퀀스:5, 입력:4
입력 데이터(4차원) 5개가 하나의 시퀀스로 묶임(sequence_length = 5)
각 배치별로 처리하는 sequence_length를 다르게 할 수 있음. 각 배치별로 순서대로 5, 3, 4개 시퀀스를 입력
'''
with tf.variable_scope('batches_dynamic_length') as scope:
    # One cell RNN input_dim (4) -> output_dim (5). sequence: 5, batch 3
    # 3 batches 'hello', 'eolll', 'lleel'
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)

    hidden_size = 2
    cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, sequence_length=[5, 3, 4], dtype=tf.float32)
    # batch가 3인데, 각 batch에 대하여 입력 시퀀스 크기를 정한 것.
    # 즉, 첫 batch는 5개 입력, 두번째 batch는 3개 입력, 마지막 batch는 4개 입력하여 수행함.
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

'''
출력: (3배치, 5길이시퀀스, 2차원출력) 
세포: (3배치, 2히든) 상태
입력: (3배치, 5길이시퀀스, 4차원입력)
'''
with tf.variable_scope('initial_state') as scope:
    batch_size = 3
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)

    # One cell RNN input_dim (4) -> output_dim (5). sequence: 5, batch: 3
    hidden_size = 2
    cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data,
                                         initial_state=initial_state, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

batch_size=3
sequence_length=5
input_dim=3

x_data = np.arange(45, dtype=np.float32).reshape(batch_size, sequence_length, input_dim)
pp.pprint(x_data)  # batch, sequence_length, input_dim

'''
출력: (3배치, 5시퀀스길이, 5차원출력) 
세포: (3배치, 5히든) 상태
입력: (3배치, 5시퀀스길이, 3입력차원)
'''
with tf.variable_scope('generated_data') as scope:
    # One cell RNN input_dim (3) -> output_dim (5). sequence: 5, batch: 3
    cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, # (3, 5,  3)
                                         initial_state=initial_state, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

'''
3개의 셀
'''
with tf.variable_scope('MultiRNNCell') as scope:
    # Make rnn
    cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
    cell = rnn.MultiRNNCell([cell] * 3, state_is_tuple=True) # 3 layers

    # rnn in/out
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    print("dynamic rnn: ", outputs)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())  # batch size, unrolling (time), hidden_size

'''

'''
with tf.variable_scope('dynamic_rnn') as scope:
    cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32, sequence_length=[1, 3, 2])
    # seq length 1 for batch 1, length 3 for batch 2, 2 for 3

    print("dynamic rnn: ", outputs)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())  # batch size, unrolling (time), hidden_size

'''
오, 양방향!!
'''
with tf.variable_scope('bi-directional') as scope:
    # bi-directional rnn
    cell_fw = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
    cell_bw = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_data, sequence_length=[2, 3, 1], dtype=tf.float32)

    sess.run(tf.global_variables_initializer())
    pp.pprint(sess.run(outputs))
    pp.pprint(sess.run(states))

# flattern based softmax
hidden_size=3
sequence_length=5
batch_size=3
num_classes=5

pp.pprint(x_data) # hidden_size=3, sequence_length=4, batch_size=2
x_data = x_data.reshape(-1, hidden_size)
pp.pprint(x_data)

softmax_w = np.arange(15, dtype=np.float32).reshape(hidden_size, num_classes)
outputs = np.matmul(x_data, softmax_w)
outputs = outputs.reshape(-1, sequence_length, num_classes) # batch, seq, class
pp.pprint(outputs)

# [batch_size, sequence_length]
y_data = tf.constant([[1, 1, 1]])

# [batch_size, sequence_length, emb_dim ]
prediction = tf.constant([[[0, 1], [1, 0], [0, 1]]], dtype=tf.float32)

# [batch_size * sequence_length]
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss = tf.contrib.seq2seq.sequence_loss(prediction, y_data, weights)
sess.run(tf.global_variables_initializer())
print("Loss: ", sequence_loss.eval())

# [batch_size, sequence_length]
y_data = tf.constant([[1, 1, 1]])

# [batch_size, sequence_length, emb_dim ]
prediction1 = tf.constant([[[0, 1], [0, 1], [0, 1]]], dtype=tf.float32)
prediction2 = tf.constant([[[1, 0], [1, 0], [1, 0]]], dtype=tf.float32)
prediction3 = tf.constant([[[0, 1], [1, 0], [0, 1]]], dtype=tf.float32)

# [batch_size * sequence_length]
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss1 = tf.contrib.seq2seq.sequence_loss(prediction1, y_data, weights)
sequence_loss2 = tf.contrib.seq2seq.sequence_loss(prediction2, y_data, weights)
sequence_loss3 = tf.contrib.seq2seq.sequence_loss(prediction3, y_data, weights)

sess.run(tf.global_variables_initializer())
print("Loss1: ", sequence_loss1.eval(),
      "Loss2: ", sequence_loss2.eval(),
      "Loss3: ", sequence_loss3.eval())

