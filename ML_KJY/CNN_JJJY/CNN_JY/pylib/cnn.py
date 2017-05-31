import tensorflow as tf
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from abc import abstractmethod
import mytool
import matplotlib.pyplot as plt
from myplot import MyPlot


class CNN:
    X = None
    X_2d = None
    Y = None

    hypothesis = None
    cost_function = None
    optimizer = None

    sess = None

    costs = []
    weights = []
    biases = []
    logs = []

    @abstractmethod
    def init_network(self):
        pass

    def set_placeholder(self, input_size, num_of_class, size_x, size_y):
        self.X = tf.placeholder(tf.float32, [None, input_size])
        self.X_2d = tf.reshape(self.X, [-1, size_x, size_y, 1])   # -1은 여러개의 입력을 의미. img 28x28x1 (black/white)
        self.Y = tf.placeholder(tf.float32, [None, num_of_class])

    #컨볼루션 층을 만듦
    def convolution_layer(self, pre_output, filter_x, filter_y, depth, num_of_filter, move_right, move_down):
        # 필터를 32개 만듦 : 3x3x1 짜리 필터
        # L1 ImgIn shape=(?, 28, 28, 1)
        W = tf.Variable(tf.random_normal([filter_x, filter_y, depth, num_of_filter], stddev=0.01)) # 3x3x1 필터 32개
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)

        # 필터 하나를 이용하여 오른쪽으로, 아래로, 가면서 convolution layer plane 하나를 만듦.
        # 모든 필터 32개를 이용하여 plane 32개를 만듦.
        # 결국  convolution layer L1은 32개 plane을 갖는 convolution layer 층 (activation maps라고 불림-32개의 28*28*1 plane)
        conv_layer = tf.nn.conv2d(pre_output, W, strides=[1, move_right, move_down, 1], padding='SAME') #오른쪽으로 1, 아래로 1
        return conv_layer

    # 기존의 층에 relu 만 적용됨
    def relu(self, layer):
        layer = tf.nn.relu(layer) # plane에 있는 각 값에 대하여 relu 적용 -> 모든 plane에 대하여 적용
        return layer

    # 풀링하여 새로운 층을 만듦
    def max_pooling(self, layer, kernel_x, kernel_y, move_right, move_down):
        # 2x2 윈도우를 오른쪽으로 2, 아래쪽으로 2씩 움직이면서 윈도우 내에 있는 가장 큰 값을 꺼내어 Pooling layer 만듦.
        mp_layer = tf.nn.max_pool(layer, ksize=[1, kernel_x, kernel_y, 1], strides=[1, move_right, move_down, 1], padding='SAME')
        # 14x14x32 풀링 레이어
        return mp_layer

    def fully_connected_layer(self, pre_layer, input_size, output_size, w_name):
        #reshaped_input = tf.reshape(pre_layer, [-1, input_size])

        # Final FC 7x7x64 inputs -> 10 outputs
        W = tf.get_variable(w_name, shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.random_normal([output_size]))
        output = tf.matmul(pre_layer, W) + b
        return output

    def set_hypothesis(self, hy):
        self.hypothesis = hy

    def set_cost_function(self):
        # define cost/loss & optimizer
        self.cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.hypothesis, labels = self.Y))

    def set_optimizer(self, l_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self.cost_function)

