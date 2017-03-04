import tensorflow as tf
import matplotlib.pyplot as plt


class LogisticClassification:

    x_data = None
    y_data = None
    W_val = []
    cost_val = []
    X_val = []
    Y_val = []
    X = None
    Y = None
    W = None
    b = None
    hypothesis = None
    cost = None
    learning_rate=None
    sess= None



    def __init__(self,x_data,y_data):
        y_len = len(y_data)
        x_len = len(x_data)
        self.x_data = x_data
        self.y_data = y_data
        soft_max = False
        try:
            x = x_data[0][0]
            self.X = tf.placeholder(tf.float32,[None, y_len])
            self.Y = tf.placeholder(tf.float32, [None, y_len])
            soft_max = True
        except:
            ''''''
            self.X = tf.placeholder(tf.float32)
            self.Y = tf.placeholder(tf.float32)
            y_len = 1
            x_len = 1
        self.sess = tf.Session()
        self.b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))


        if soft_max :
            self.W = tf.Variable(tf.random_uniform([y_len, x_len], -1.0, 1.0))
            self.hypothesis = tf.nn.softmax(tf.matmul(-self.W, self.X-self.b))
            self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y*tf.log(self.hypothesis),reduction_indices=1))
        else :
            self.W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
            self.hypothesis = tf.div(1.,1.+tf.exp(-self.W * (self.X-self.b)))
            self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis)+ (1-self.Y)*tf.log(1-self.hypothesis))


        init = tf.initialize_all_variables()
        self.sess.run(init)

    def training(self, learning_rate=0.04, step=2001, show_training_data=False):
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.cost)

        for step in range(step):
            self.sess.run(self.train, feed_dict={self.X: self.x_data, self.Y: self.y_data})
            self.W_val.append(self.sess.run(self.W, feed_dict={self.X: self.x_data}))
            self.cost_val.append(self.sess.run(self.cost, feed_dict={self.X: self.x_data, self.Y: self.y_data}))
            if show_training_data==True and step % 20 == 0 :
                print(step,'weght = ',self.sess.run(self.W,feed_dict={self.X: self.x_data, self.Y: self.y_data}),'cost =',self.sess.run(self.cost,feed_dict={self.X:self.x_data,self.Y:self.x_data}))



    def show_cost_graph(self):
        try:
            plt.plot(self.W_val, self.cost_val, 'ro')
            plt.ylabel('cost')
            plt.xlabel('weight')
            plt.show()
        except:
            print('입력값이 1차원이 아닙니다.')
    # 입력값이 1차원이였을 때 입력값과 라벨을 보여준다.
    def show_singlevariable_graph(self):
        try:
            plt.plot(self.x_data, self.y_data, 'ro')
            plt.plot(self.x_data, self.sess.run(tf.div(1.,1.+tf.exp(-self.W  * (self.x_data-self.b)))), label='fitted line')
            plt.ylabel('hypothesis')
            plt.xlabel('X')
            plt.legend()
            plt.show()
        except:
            print('입력값이 1차원이 아닙니다.')

    # 값을 넣엇을 때 라벨이 무엇이 나오나 보여준다.
    def predict(self, x_data):
        self.Y_val = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        self.X_val = x_data
        result = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        print(result)
   #     try:
        plt.plot(self.X_val, self.Y_val, 'ro')
        plt.plot(self.x_data,self.sess.run(tf.div(1.,1.+tf.exp(-self.W  * (self.x_data-self.b)))), label='fitted line')
        plt.ylabel('hypothesis')
        plt.xlabel('X')
        plt.legend()
        plt.show()
        # except:
        #     print('입력값이 1차원이 아닙니다.')