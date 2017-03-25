import tensorflow as tf
import numpy as np

class LogisticClassification:
    # 멤버변수
    # Variables
    # Data number
    x_num = None;
    y_num = None;
    w_num = None;
    data_num = None;
    # For Data
    X_training = None
    Y_training = None
    X_testing = None
    Y_testing = None
    # For hypothesis
    W = None
    X = None
    Y = None
    hypothesis = None
    cost = None
    # Session
    sess = None

    # 생성자함수. 초기화.
    def __init__(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, None], name="x_ph")
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, ], name="y_ph")

        self.sess = tf.Session()

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

        self.w_num = len(self.X_training)
        self.x_num = len(self.X_training)
        self.y_num = len(self.Y_training)

        # Multi variable 판별
        if len(self.X_training) == 2: self.IsMulti = False
        else: self.IsMulti = True

    # Learn
    # Linear Regression의 학습과정 (w, b 찾기)
    # cost값이 finish_point 이하면 종료
    # gildong.learn(0.001)와 같이 사용
    def learn(self, finish_point):

        self.W = tf.Variable(tf.random_uniform([1, self.w_num], -1.0, 1.0))

        # Our hypothesis
        h = tf.matmul(self.W, self.X)  # 이전 hypothesis
        self.hypothesis = tf.sigmoid(h)

        # cost function
        self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis) + (1 - self.Y) * tf.log(1 - self.hypothesis))

        # Minimize
        a = tf.Variable(0.5)  # Learning rate
        optimizer = tf.train.GradientDescentOptimizer(a)
        train = optimizer.minimize(self.cost)

        # initialize the variables
        init = tf.global_variables_initializer()

        # Launch the graph.
        self.sess.run(init)

        step = 0
        while True:
            self.sess.run(train, feed_dict={self.X: self.X_training, self.Y: self.Y_training})
            step += 1
            if self.sess.run(self.cost, feed_dict={self.X: self.X_training, self.Y: self.Y_training}) < finish_point : # cost값이 일정 이하로 내려가면 함수 종료
                break

    # test 1
    # testing set을 이용한 학습 결과 test
    # gildong.test()와 같이 사용
    def test(self):
        prediction = self.sess.run(self.hypothesis, feed_dict={self.X: self.X_testing}) > 0.5
        label = self.Y_testing > 0.5

        if prediction.all() == label.all():
            print ("학습 success")
        else:
            print ("학습 fail")

    # 실제 수행 함수
    # logistic_classification() 사용 후 what_is_it() 사용
    def logistic_classification(self, txt_file_name):
        print ("logistic classification")
        self.set_data(txt_file_name)
        print("learn..")
        self.learn(0.1)
        self.test()

    # prediction
    # 학습결과를 토대로 예측
    # 매개변수로 x 배열(혹은 x)을 받음
    # gildong.what_is_it([3,4])와 같이 사용
    def what_is_it(self, input_data):
        # data input
        x_data = np.ones((self.x_num, 1))

        for i in range(0, self.x_num - 1):
            x_data[i+1, 0] = input_data[i]

        # prediction output
        print ("prediction : ",  self.sess.run(self.hypothesis, feed_dict={self.X:x_data}) > 0.5)