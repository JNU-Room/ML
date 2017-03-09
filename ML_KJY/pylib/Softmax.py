import tensorflow as tf

from .Transposition import Trasposition

class Softmax:

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
        self.x_data = x_data
        self.y_data = y_data

            # x = x_data[0][0]
        self.X = tf.placeholder(tf.float32,[None, None])
        self.Y = tf.placeholder(tf.float32, [None , None])
        self.sess = tf.Session()
        self.b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

        self.W = tf.Variable(tf.random_uniform([len(x_data),len(y_data)], -1.0, 1.0))
                                                    #4,4        4,4
        self.hypothesis = tf.nn.softmax(tf.matmul(self.X-self.b,-self.W))
                                                  #3,5         3,5
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y*tf.log(self.hypothesis),reduction_indices=1))
        # else :
        #     self.W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        #     self.hypothesis = tf.div(1.,1.+tf.exp(-self.W * (self.X-self.b)))
        #     self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis)+ (1-self.Y)*tf.log(1-self.hypothesis))


        init = tf.initialize_all_variables()
        self.sess.run(init)

    def training(self, learning_rate=0.04, step=2001, show_training_data=False):
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.cost)

        for step in range(step):
            self.sess.run(self.train, feed_dict={self.X: self.x_data, self.Y: self.y_data})
            self.W_val.append(self.sess.run(self.W, feed_dict={self.X: self.x_data}))
            self.cost_val.append(self.sess.run(self.cost, feed_dict={self.X: self.x_data, self.Y: self.y_data}))
            if show_training_data==True and step % 200 == 0 :
                print(step,'weght = ',self.sess.run(self.W,feed_dict={self.X: self.x_data, self.Y: self.y_data}),'cost =',self.sess.run(self.cost,feed_dict={self.X:self.x_data,self.Y:self.y_data}))



    def predict(self, x_data):
        self.Y_val = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        self.X_val = x_data

        result = self.sess.run(self.hypothesis, feed_dict={self.X:x_data })


        print(self.sess.run(self.W))
        print(result)
        print(self.sess.run(tf.argmax(result,1)))
