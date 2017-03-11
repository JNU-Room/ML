import tensorflow as tf
import matplotlib.pyplot as plt

#LenearRegession 다중 변수 또는 단일변수 둘다 사용 가능
#다중 변수로 들어가는 경우 x[0][0],x[1][0],x[2][0]... 으로 동시에 들어가게 된다
#x_data의 shape는  (feature의 개수  , 입력데이터집합의 개수)또는 1차원인경우 (입력데이터집합의 개수)
#y_data의 shape는 1차원이 나오고 (예측데이터 집합의 개수 =입력데이터 집합의 개수)


class LenearRegression:
	W_val = []
	cost_val = []
	X_val = []
	Y_val = []
	sess =None
	hypothesis = None
	W= None
	b= None
	x_data = None
	y_data = None
	optimizer=None
	train = None
	cost = None
	X = None
	Y = None

#생성자 함수로 LenearRegession에서 반드시 필요한 부분 처리
	def __init__(self,x_data,y_data):
		multivariable = False
		a = x_data
		self.sess = tf.Session()
		self.X = tf.placeholder(tf.float32)
		self.Y = tf.placeholder(tf.float32)
		self.x_data = x_data
		self.y_data = y_data

		#x_data가 다차원리스트인지 확인
		try:
			x = a[0][0]
			multivariable = True
		except:
			''''''

		self.b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
		if multivariable:
			self.W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))
			self.hypothesis = tf.matmul(self.W,self.X) + self.b
		else:
			self.W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
			self.hypothesis = self.X*self.W + self.b


		self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y))



		init = tf.initialize_all_variables()
		self.sess.run(init)

	#학습시키는 함수
	def training(self,learning_rate=0.04, step = 2001,show_training_data=True):
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.train = self.optimizer.minimize(self.cost)

		for step in range(step):
			self.sess.run(self.train,feed_dict={self.X:self.x_data,self.Y:self.y_data})
			self.W_val.append(self.sess.run(self.W,feed_dict={self.X:self.x_data}))
			self.cost_val.append(self.sess.run(self.cost,feed_dict={self.X:self.x_data,self.Y:self.y_data}))
			if show_training_data == True and step % 20 == 0:
				print(step, 'weght = ', self.sess.run(self.W, feed_dict={self.X: self.x_data, self.Y: self.y_data}),
					  'cost =', self.sess.run(self.cost, feed_dict={self.X: self.x_data, self.Y: self.y_data}))


	#입력값이 1차원이였을 때 코스트함수를 보여준다.
	def show_cost_graph(self):
		try:
			plt.plot(self.W_val, self.cost_val,'ro')
			plt.ylabel('cost')
			plt.xlabel('weight')
			plt.show()
		except:
			print('입력값이 1차원이 아닙니다.')

	#입력값이 1차원이였을 때 입력값과 라벨을 보여준다.
	def show_singlevariable_graph(self):
		try:
			plt.plot(self.x_data, self.y_data,'ro')
			plt.plot(self.x_data,self.sess.run(self.W) * self.x_data + self.sess.run(self.b),label = 'fitted line')
			plt.ylabel('hypothesis')
			plt.xlabel('X')
			plt.legend()
			plt.show()
		except:
			print('입력값이 1차원이 아닙니다.')

	# 조건에 맞는 입력 데이타를 받으면 회귀에 따라 예측이되는 출력값을 보낸다
	def predict(self,x_data):

		self.Y_val = self.sess.run(self.hypothesis, feed_dict={self.X: x_data, self.Y: self.y_data})
		self.X_val = x_data

		print(self.sess.run(self.hypothesis,feed_dict={self.X:x_data}))
		try:
			plt.plot(self.X_val, self.Y_val, 'ro')
			plt.plot(self.x_data, self.sess.run(self.W) * self.x_data + self.sess.run(self.b), label='fitted line')
			plt.ylabel('hypothesis')
			plt.xlabel('X')
			plt.legend()
			plt.show()
		except:
			print('입력값이 1차원이 아닙니다.')

		

	
