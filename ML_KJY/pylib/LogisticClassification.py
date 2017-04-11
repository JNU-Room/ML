import tensorflow as tf
import matplotlib.pyplot as plt

#LogisticClassification 다중 변수 또는 단일변수 둘다 사용 가능
#입력값은 x[i]가 통채로 하나씩 들어간다.
#x_data의 shape는  (데이터의 수  , feature의 수)또는 1차원인경우 (입력데이터집합의 개수)
#y_data의 shape는 1차원이 나오고 (예측데이터 집합의 개수 =입력데이터 집합의 개수)
#y_data학습할 때는 반드시 0또는 1이여야 한다.
class LogisticClassification:

    x_data = None
    y_data = None
    W_val = []
    cost_val = []
    X_val = []
    Y_val = []
    list_step = []
    X = None
    Y = None
    weights = []
    bias = []
    hypothesis = None
    cost = None
    learning_rate=None
    sess= None
    layer = 0


    def __init__(self,x_data,y_data):
        self.one_D = True
        self.x_data = x_data
        self.y_data = y_data

        #rank=1인지 2인지 판단

        x = x_data[0][0]
        self.X = tf.placeholder(tf.float32,[None, None])
        self.Y = tf.placeholder(tf.float32, [None, None])


        self.sess = tf.Session()
        #rank가 1인지 2인지에따라 가설과 weight 형태 조정


    def create_layer(self, X, input_length, output_length):
        self.weights.append(tf.Variable(tf.random_uniform([input_length, output_length],-1.0,1.0)))
        self.bias.append(tf.Variable(tf.random_uniform([output_length], -1.0, 1.0)))

        ret = tf.add(tf.matmul(X, self.weights[-1]),self.bias[-1])
        self.layer += 1
        ret = tf.nn.relu(ret)
        return ret


    def set_cost(self, input_data, input_length):
        self.bias.append(tf.Variable(tf.random_uniform([len(self.y_data[0])], -1.0, 1.0)))
        self.weights.append(tf.Variable(tf.random_uniform([input_length, len(self.y_data[0])], -1.0, 1.0)))
        self.hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(input_data, -self.weights[-1] ) + self.bias[-1]))
        self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis) + (1 - self.Y) * tf.log(1 - self.hypothesis))

        init = tf.initialize_all_variables()
        self.sess.run(init)


    def training(self, learning_rate=0.04, step=2001, show_training_data=False):
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.cost)
        self.list_step = range(step)
        for step in range(step):
            self.sess.run(self.train, feed_dict={self.X: self.x_data, self.Y: self.y_data})
            self.cost_val.append(self.sess.run(self.cost, feed_dict={self.X: self.x_data, self.Y: self.y_data}))
            if show_training_data==True and step % 20 == 0 :
                print(step,'weght = ',self.sess.run(self.weights[-1],feed_dict={self.X: self.x_data, self.Y: self.y_data}),
                            'cost =',self.sess.run(self.cost,feed_dict={self.X:self.x_data,self.Y:self.y_data}))

        plt.plot(self.list_step, self.cost_val)
        plt.ylabel('cost')
        plt.xlabel('step')

    def show_cost_graph(self):
        plt.plot(self.list_step, self.cost_val)
        plt.ylabel('cost')
        plt.xlabel('step')
        plt.show()
    # 입력값이 1차원이였을 때 입력값과 라벨을 보여준다.
    # def show_singlevariable_graph(self):
    #     try:
    #         plt.plot(self.x_data, self.y_data, 'ro')
    #         plt.plot(self.x_data, self.sess.run(tf.div(1.,1.+tf.exp(-self.W  * (self.x_data-self.b)))), label='fitted line')
    #         plt.ylabel('hypothesis')
    #         plt.xlabel('X')
    #         plt.legend()
    #         plt.show()
    #     except:
    #         print('입력값이 1차원이 아닙니다.')

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

