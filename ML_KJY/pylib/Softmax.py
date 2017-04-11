import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#Softmax 단일변수는 의미가 없기때문에 다중 변수만 사용 가능
#입력값은 x[i]가 통채로 하나씩 들어간다.
#x_data의 shape는  (입력데이터의 집합개수 , feature의 개수)
#y_data의 shape는 (feature의 개수 , 신경의 수)
#y_data학습할때는 반드시 0또는 1이여야 한다.
class Softmax:

    x_data = None
    y_data = None
    W_val = []
    cost_val = []
    X_val = []
    Y_val = []
    X = None
    Y = None
    weights = []
    bias = []
    hypothesis = None
    cost = None
    learning_rate=None
    sess= None
    result = None
    layer = 0
    list_step = 0

    def __init__(self,x_data,y_data):
        self.x_data = x_data
        self.y_data = y_data


        self.X = tf.placeholder(tf.float32,[None, None])
        self.Y = tf.placeholder(tf.float32, [None , None])
        self.sess = tf.Session()


    #코스트를 만드는데 필요한 weight, bias, hypothesis 모두 정의 함수이름 생각나면 변경하기
    def set_cost(self,input_data,input_length):
        self.bias.append(tf.Variable(tf.random_uniform([len(self.y_data[0])], -1.0, 1.0)))
        self.weights.append(tf.Variable(tf.random_uniform([input_length,len(self.y_data[0])], -1.0, 1.0)))
        self.hypothesis = tf.nn.softmax(tf.add(tf.matmul(input_data,-self.weights[-1]),self.bias[-1]))
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y*tf.log(self.hypothesis),reduction_indices=1))
        init = tf.initialize_all_variables()
        self.sess.run(init)


    def training(self, learning_rate=0.04, step=2001, show_training_data=False):
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.cost)
        self.list_step = range(step)
        for step in range(step):
            self.sess.run(self.train, feed_dict={self.X: self.x_data, self.Y: self.y_data})
            self.cost_val.append(self.sess.run(self.cost, feed_dict={self.X: self.x_data, self.Y: self.y_data}))
            if show_training_data==True and step % 400 == 0 :
                print(step,'weght = ',self.sess.run(self.weights,feed_dict={self.X: self.x_data, self.Y: self.y_data}),'cost =',self.sess.run(self.cost,feed_dict={self.X:self.x_data,self.Y:self.y_data}))

    def show_cost_graph(self):
        plt.plot(self.list_step, self.cost_val)
        plt.ylabel('cost')
        plt.xlabel('step')
        plt.show()
    #조건에 맞는 입력 데이타를 받으면 회귀에 따라 예측이되는 출력값을 보낸다
    def predict(self, x_data):
        # self.Y_val = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        # self.X_val = x_data
        temp = x_data
        for i in range(len(self.weights)-1):
            temp = tf.matmul(temp,self.weights[i])

        self.hypothesis = tf.nn.softmax(tf.add(tf.matmul(self.X, -self.weights[-1]), self.bias[-1]))
        try:
            self.result = self.sess.run(self.hypothesis, feed_dict={self.X:self.sess.run(temp) })
        except:
            self.result = self.sess.run(self.hypothesis, feed_dict={self.X:temp})

        print(self.sess.run(tf.argmax(self.result,1))) #argmax가 one-hot encoding
        return self.sess.run(tf.argmax(self.result,1))
    def return_predict_possibility(self):
        return self.result

    def return_predict_onehot(self):
        return self.sess.run(tf.argmax(self.result, 1))

    def save_weight(self):
        np.save('weight',self.sess.run(self.weights))
        np.save('bias',self.sess.run(self.bias))

    def load_weight(self):
        self.weights = np.load('weight.npy')
        self.bias = np.load('bias.npy')


    def create_layer(self, X, input_length, output_length):
        self.weights.append(tf.Variable(tf.random_uniform([input_length, output_length],-1.0,1.0)))
        self.bias.append(tf.Variable(tf.random_uniform([output_length], -1.0, 1.0)))
        ret = tf.add(tf.matmul(self.X,self.weights[self.layer]),self.bias[-1])
        self.layer += 1
        ret = tf.nn.relu(ret)
        return ret