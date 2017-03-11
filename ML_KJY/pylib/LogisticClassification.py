import tensorflow as tf
import matplotlib.pyplot as plt

#LogisticClassification 다중 변수 또는 단일변수 둘다 사용 가능
#다중 변수로 들어가는 경우 x[0][0],x[1][0],x[2][0]... 으로 동시에 들어가게 된다
#x_data의 shape는  (입력데이터의 집합개수 , feature의 개수)
#y_data의 shape는 (feature의 개수 , weight의 shape중 두번째 값)
#y_data학습할때는 반드시 0또는 1이여야 한다.
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
        one_D = True
        self.x_data = x_data
        self.y_data = y_data

        #rank=1인지 2인지 판단
        try:
            x = x_data[0][0]
            self.X = tf.placeholder(tf.float32,[None, None])
            self.Y = tf.placeholder(tf.float32)
            one_D = False
        except:
            ''''''
            self.X = tf.placeholder(tf.float32)
            self.Y = tf.placeholder(tf.float32)


        self.sess = tf.Session()
        self.b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

        #rank가 1인지 2인지에따라 가설과 weight 형태 조정
        if one_D :
            self.W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
            self.hypothesis = tf.div(1.,1.+tf.exp(-self.W * (self.X-self.b)))
        else :
            self.W = tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, 1.0))
            self.hypothesis = tf.div(1.,1.+tf.exp(tf.matmul(-self.W , (self.X-self.b))))

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
                print(step,'weght = ',self.sess.run(self.W,feed_dict={self.X: self.x_data, self.Y: self.y_data}),'cost =',self.sess.run(self.cost,feed_dict={self.X:self.x_data,self.Y:self.y_data}))



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

    # 입력값을 형식에 맞게 넣은 경우 회귀에 따른 예측값을 보여준다.
    def predict(self, x_data):
        self.Y_val = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        self.X_val = x_data
        result = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        print(result)

        try:
            plt.plot(self.X_val, self.Y_val, 'ro')
            plt.plot(self.x_data,self.sess.run(tf.div(1.,1.+tf.exp(-self.W  * (self.x_data-self.b)))), label='fitted line')
            plt.ylabel('hypothesis')
            plt.xlabel('X')
            plt.legend()
            plt.show()
        except:
             print('입력값이 1차원이 아닙니다.')