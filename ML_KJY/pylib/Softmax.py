import tensorflow as tf

#Softmax 단일변수는 의미가 없기때문에 다중 변수만 사용 가능
#다중 변수로 들어가는 경우 선형회귀나 분류와는 다르게 x[0][0],x[0][1],x[0][2]... 으로 동시에 들어가게 된다
#x_data의 shape는  (입력데이터의 집합개수 , feature의 개수)
#y_data의 shape는 (feature의 개수 , weight의 shape중 두번째 값)
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
    W = None
    b = None
    hypothesis = None
    cost = None
    learning_rate=None
    sess= None

    def __init__(self,x_data,y_data):
        self.x_data = x_data
        self.y_data = y_data


        self.X = tf.placeholder(tf.float32,[None, None])
        self.Y = tf.placeholder(tf.float32, [None , None])
        self.sess = tf.Session()
        self.b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

        self.W = tf.Variable(tf.random_uniform([len(x_data[0]),len(y_data[0])], -1.0, 1.0))
        self.hypothesis = tf.nn.softmax(tf.matmul(self.X-self.b,-self.W))
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y*tf.log(self.hypothesis),reduction_indices=1))

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


    #조건에 맞는 입력 데이타를 받으면 회귀에 따라 예측이되는 출력값을 보낸다
    def predict(self, x_data):
        self.Y_val = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        self.X_val = x_data

        result = self.sess.run(self.hypothesis, feed_dict={self.X:x_data })

        #print(self.sess.run(self.W))
        print(result)
        print(self.sess.run(tf.argmax(result,1))) #argmax가 one-hot encoding
        #print(self.rank(list(result)))


    # def rank(self, List):
    #     i = 0
    #     print(List)
    #     Listy = List[:]
    #     Listx = List[:]
    #     Listx.sort()
    #     print(Listx)
    #     for value in Listx:
    #         Listy[i] = Listx.whare(Listx == value)
    #     x_len = len(xxx);
    #         i += 1
    #     return Listy