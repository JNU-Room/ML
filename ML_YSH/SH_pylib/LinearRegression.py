import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    # Variables
    # For Data
    X_training = 0.
    Y_training = 0.
    X_testing = 0.
    Y_testing = 0.
    # For hypothesis
    W = None
    X = tf.placeholder(dtype=tf.float32, shape=[None, None])
    #b = None # only 1-variable
    hypothesis = 0
    # select
    IsMulti = False

    # Session
    sess = tf.Session()

    # Set data
    # txt 파일명을 받으면, 데이터를 입력받는다.
    # gildong.set_data('file.txt')와 같이 사용
    def set_data(self, txt_file_name):
        # load txt
        xy = np.loadtxt(txt_file_name, unpack=True, dtype='float32')
        data_num = len(xy[0, :])

        # Training set
        training_num = int(data_num * (4/5)) # 전체 data의 80%만 train

        self.X_training = xy[0:-1, 0:training_num]
        self.Y_training = xy[-1, 0:training_num]

        # Testing set
        self.X_testing = xy[0:-1, training_num:] # training data가 아닌 data
        self.Y_testing = xy[-1, training_num:]

        # Multi variable 판별
        if len(self.X_training) == 2: self.IsMulti = False
        else: self.IsMulti = True

    # Learn
    # Linear Regression의 학습과정 (w, b 찾기)
    # cost값이 finish_point 이하면 종료
    # gildong.learn(0.001)와 같이 사용
    def learn(self, finish_point):
        # w 배열 크기 구하기 (b를 포함)
        w_num = len(self.X_training)

        # Try to find values for W and b that compute y_data = W * x_data + b
        self.W = tf.Variable(tf.random_uniform([1, w_num], -1.0, 1.0))  # 1 * w_num 행렬 (b를 포함)
        #self.W = np.random.rand(-1, 1, (1, w_num))
        # without b # b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

        # Our hypothesis
        self.hypothesis = tf.matmul(self.W, self.X)  # 행렬곱

        # Simplified cost function
        cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y_training))

        # Minimize
        a = tf.Variable(0.1)  # Learning rate
        optimizer = tf.train.GradientDescentOptimizer(a)
        train = optimizer.minimize(cost)

        # initialize the variables
        init = tf.global_variables_initializer()

        # Launch the graph.
        self.sess.run(init)

        # Fit the line
        step = 0
        while 1:
            self.sess.run(train, feed_dict={self.X:self.X_training})
            step += 1
            if step == 1000:
                break
            if self.sess.run(cost, feed_dict={self.X:self.X_training}) < finish_point : # cost값이 일정 이하로 내려가면 함수 종료
                break

    # Output
    # 학습된 Linear Regession hypothesis 출력
    # 0 : hypothesis 정보 출력, 1 : hypothesis 그래프 출력
    # gildong.show_wb(1)과 같이 사용
    def show_wb(self, select):
        # For graph
        x_val = []
        y_val = []
        # only one-variable
        w = self.W[0, 1]
        b = self.W[0, 0]
        x_one = tf.placeholder(dtype=tf.float32)
        hyp_one = tf.mul(w,x_one)+b

        # initialize the variables
        init = tf.global_variables_initializer()

        # Launch the graph.

        if select == 0: # W 배열 출력
            if self.IsMulti == True: # multi-variable
                print ("W : ", self.sess.run(self.W))
                print ("hypothesis :" , self.sess.run(self.W) , " * X" )
            else: # 1-variable
                print ("W : ", self.sess.run(w), "b : ", self.sess.run(b))
                print ("hypothesis : ", self.sess.run(w), " * X + ", self.sess.run(b))
        elif select == 1: # hypothesis 그래프 출력
            if self.IsMulti == True: # multi-variable
                print ("다차원 그래프는 출력할 수 없습니다.")
            else: # one-variable
                for i in range(-30, 50):
                    x_val.append(0.1 * i)
                    y_val.append(self.sess.run(hyp_one, feed_dict={x_one: i * 0.1}))
                # Graphic display
                plt.plot(x_val, y_val, 'ro')
                plt.ylabel('y')
                plt.xlabel('x')
                plt.show()
    # test 1
    # testing set을 이용한 학습 결과 test
    # gildong.test()와 같이 사용
    def test(self):
        prediction = self.sess.run(self.hypothesis, feed_dict={self.X: self.X_testing})
        label = self.Y_testing
        if (label-prediction) < 0.01:
            print ("학습 success")
        else:
            print ("학습 fail")

    # prediction
    # 학습결과를 토대로 예측
    # 매개변수로 x 배열(혹은 x)을 받음
    # gildong.what_is_it([3,4])와 같이 사용
    def what_is_it(self, input_data):
        # data input
        x_data = np.ones((len(self.X_training), 1))
        x_data[1:] = input_data

        # prediction output
        print ("prediction : ",  self.sess.run(self.hypothesis, feed_dict={self.X:x_data}))

    # 실제 수행 함수
    # leaner_regression() 사용 후 what_is_it() 사용
    def linear_regression(self, txt_file_name):
        print ("linear regression")
        self.set_data(txt_file_name)
        print("learn..")
        self.learn(0.000001)
        # print("show wb")
        # self.show_wb(1)\
        self.test()
