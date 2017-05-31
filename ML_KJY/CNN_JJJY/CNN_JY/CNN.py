# Lab 10 MNIST and Dropout
import tensorflow as tf
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import matplotlib.pyplot as plt

class CNN():
    # tf.set_random_seed(777)  # reproducibility
    W_val = []
    cost_val = []
    X_val = []
    Y_val = []
    list_step = []
    sess = None
    hypothesis = None
    weights = []
    bias = []
    x_data = None
    y_data = None
    optimizer = None
    train = None
    cost = None
    X = None
    Xx = None
    Y = None
    layer = 0
    keep_prob = 0.7
    def __init__(self, x_data, y_data):
        self.sess = tf.Session()
        self.X = tf.placeholder(tf.float32, [None ,None, None, None])
        self.Y = tf.placeholder(tf.float32, [None, None])
        self.Xx = tf.placeholder(tf.float32, [None, None])
        self.x_data = x_data
        self.y_data = y_data

    # parameters
    learning_rate = 0.001
    training_epochs = 10 #15
    batch_size = 100

    def convolution_layer(self, pre_output, filter_x, filter_y, depth, num_of_filter, move_right, move_down):

        self.weights.append(tf.Variable(tf.random_normal([filter_x, filter_y, depth, num_of_filter], stddev=0.01)))
        self.bias.append(tf.Variable(tf.random_normal([num_of_filter])))
        conv_layer = tf.nn.conv2d(pre_output, self.weights[-1], strides=[1, move_right, move_down, 1], padding='SAME') #오른쪽으로 1, 아래로 1
        return conv_layer

    def final_layer(self, input, height, witdth):
        Layer1 = tf.reshape(input, [-1, height * witdth * 5 ])
        print(Layer1)
        # L4 FC 4x4x128 inputs -> 625 outputs
        self.W = tf.Variable(tf.random_uniform([height * witdth * 5, 625], -1.0, 1.0))
            # tf.get_variable("W4", shape=[188 * 360 * 5, 625],
            #                  initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.Variable(tf.random_normal([625]))
        layer2 = tf.nn.relu(tf.matmul(Layer1, self.W) + self.b)
        layer2 = tf.nn.dropout(layer2, keep_prob=self.keep_prob)
        print(layer2)
        self.weights.append(tf.Variable(tf.random_uniform([625, 2], -1.0, 1.0)))

            # tf.get_variable("W5", shape=[625, 20],
            #                  initializer=tf.contrib.layers.xavier_initializer()))
        self.bias.append(tf.Variable(tf.random_normal([2])))
        self.hypothesis = tf.matmul(self.Xx, self.weights[-1]) + self.bias[-1]
        # init = tf.initialize_all_variables()
        # self.sess.run(init)
        return  layer2



    def max_pooling(self, layer, kernel_x, kernel_y, move_right, move_down):
        # 2x2 윈도우를 오른쪽으로 2, 아래쪽으로 2씩 움직이면서 윈도우 내에 있는 가장 큰 값을 꺼내어 Pooling layer 만듦.
        mp_layer = tf.nn.max_pool(layer, ksize=[1, kernel_x, kernel_y, 1], strides=[1, move_right, move_down, 1], padding='SAME')
        # 14x14x32 풀링 레이어
        return mp_layer

    def dropout(self, Layer):
        Layer = tf.nn.dropout(Layer, keep_prob=self.keep_prob)
        return Layer

    def set_cost_function(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.hypothesis, labels=self.Y))


    def train(self, x_data, y_data, keep_prop=0.7, learning_rate = 0.1, step = 200):
        train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        init = tf.initialize_all_variables()
        self.sess.run(init)
        print(self.sess.run(self.cost, feed_dict = {self.Xx : self.sess.run(x_data), self.Y: y_data}))
        for i in range(step):
            self.sess.run(train, feed_dict={self.Xx: self.sess.run(x_data), self.Y: y_data})
            if i % 2 == 0:
                print(i, 'cost =', self.sess.run(self.cost, feed_dict={self.Xx : self.sess.run(x_data), self.Y: y_data}))
        print(self.sess.run(tf.nn.softmax(self.hypothesis), feed_dict = {self.Xx : self.sess.run(x_data)}))

    def predict(self, x_data, conv_layer_time):
        for i in range(conv_layer_time):
            x_data = tf.nn.conv2d(x_data, self.weights[i], strides=[1, 1, 1, 1], padding='SAME')
                                                                        #일단 1 고정
        print(x_data)
        x_data = tf.reshape(x_data, [-1, len(self.sess.run(x_data[0])) * len(self.sess.run(x_data[0][0])) * 5])
        x_data = tf.nn.relu(tf.matmul(x_data, self.W) + self.b)
        x_data = tf.nn.dropout(x_data, keep_prob=self.keep_prob)
        print(self.sess.run(tf.nn.softmax(self.hypothesis), feed_dict={self.Xx: self.sess.run(x_data)}))