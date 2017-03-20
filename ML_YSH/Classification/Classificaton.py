import tensorflow as tf
import numpy as np
import matplotlib as plt

class Classification:
    # Binary(Sigmoid) 또는 Multi(SoftMax)

    # Variables
    # Data number
    x_num = None;
    y_num = None;
    data_num = None;
    # For Data
    X_training = None
    Y_training = None
    X_testing = None
    Y_testing = None
    # For hypothesis
    W = None
    X = tf.placeholder(dtype=tf.float32, shape=[None, None])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, None])
    hypothesis = None
    cost = None
    # select
    IsSigmoid = None
    IsSoftMax = None
    # Session
    sess = None

    # 생성자 함수.
    # 자동 실행. 변수 초기화
    def __init__(self):
        self.IsSigmoid = False
        self.IsSoftMax = False
        self.sess = tf.Session()

    # Set data
    # txt 파일명을 받으면, 데이터를 입력받는다.
    # y_num은 분류 클래스 개수
    # gildong.set_data('file.txt', 1)와 같이 사용
    def set_data(self, txt_file_name, y_num):
        # load txt
        xy = np.loadtxt(txt_file_name, unpack=True, dtype='float32')
        self.data_num = len(xy[0, :])
        self.y_num = y_num
        self.x_num = self.data_num - self.y_num

        # Training set
        training_num = int(self.data_num * (4/5)) # 전체 data의 80%만 train

        self.X_training = xy[0:self.x_num, 0:training_num]
        self.Y_training = xy[self.x_num:, 0:training_num]

        # Testing set
        self.X_testing = xy[0:self.x_num, training_num:] # training data가 아닌 data
        self.Y_testing = xy[self.x_num:, training_num:]

        # Sigmoid , SoftMax 택
        if self.y_num < 1 : exit()
        elif self.y_num == 1: self.IsSigmoid = True
        else: self.IsSoftMax = True

    # Learn
    # 학습과정.
    # cost값이 finish_point 이하면 종료
    # gildong.learn(0.001)와 같이 사용
    def learn(self, finish_point):
        # w 배열 크기 구하기 (b를 포함)
        w_num = self

        # weight 설정
        if self.IsSigmoid == True:
            self.W = tf.Variable(tf.random_uniform([1, w_num], -1.0, 1.0))  # 1 * w_num 행렬 (b를 포함)
        else:
            self.W = tf.Variable(tf.zeros([self.x_num, self.y_num]))

        # Our hypothesis & Cost Function
        if self.IsSigmoid:
            self.hypothesis = tf.div(1., 1. + tf.exp(-tf.matmul(self.W, self.X))) # Sigmoid 적용
            self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis) + (1 - self.Y) * tf.log(1 - self.hypothesis)) # Cost
        else:
            self.hypothesis = tf.nn.softmax(tf.matmul(self.X, self.W))  # Softmax 적용
            self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.hypothesis), reduction_indices=1)) # Cross entropy

        # Minimize
        a = tf.Variable(0.01)  # Learning rate
        optimizer = tf.train.GradientDescentOptimizer(a)
        train = optimizer.minimize(self.cost)

        # initialize the variables
        init = tf.global_variables_initializer()

        # Launch the graph.
        self.sess.run(init)

        # Fit the line
        step = 0
        print("step, cost, W")
        print(step, self.sess.run(self.cost, feed_dict={self.X: self.X_training}), self.sess.run(self.W))
        while 1:
            self.sess.run(train, feed_dict={self.X: self.X_training})
            step += 1
            if step % 20 == 0:
                print(step, self.sess.run(self.cost, feed_dict={self.X: self.X_training}), self.sess.run(self.W))
            if step == 1000:
                break
            if self.sess.run(self.cost, feed_dict={self.X: self.X_training}) < finish_point:  # cost값이 일정 이하로 내려가면 함수 종료
                print(step, self.sess.run(self.cost, feed_dict={self.X: self.X_training}), self.sess.run(self.W))
                break

    # Output
    # test
    # prediction
# main
gildong = Classification()
y_num = int( input("Class 개수(y 개수) 입력") )
gildong.set_data('train.txt', y_num)
gildong.learn(0.01)

