import tensorflow as tf
import matplotlib.pyplot as plt

class LenearRegession:
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
		i=0
		a = x_data
		self.sess = tf.Session()
		self.X = tf.placeholder(tf.float32)
		self.Y = tf.placeholder(tf.float32)
		m = len(x_data)
		self.x_data = x_data
		self.y_data = y_data

		#a가 몇차원 리스트인지 알아내기 위함
		#일단 현재로서는 2차원 데이터 입력까지만 지원
		while(1):
			try:
				x = a[i]
				i += 1
				print(i,a)
			except:
				break
		if i != 1:
			self.W = tf.Variable(tf.random_uniform([1,i],-1.0,1.0))

		else:
			self.W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))



		self.b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

		try:
			self.hypothesis = tf.matmul(self.W,self.X) + self.b
		except:
			self.hypothesis = self.X*self.W + self.b

		self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y))



		init = tf.initialize_all_variables()
		self.sess.run(init)

	#학습시키는 함수
	def training(self,learning_rate=0.04, step = 2001):
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.train = self.optimizer.minimize(self.cost)

		for step in range(step):
			self.sess.run(self.train,feed_dict={self.X:self.x_data,self.Y:self.y_data})
		#	try:
			self.W_val.append(self.sess.run(self.W,feed_dict={self.X:self.x_data}))
			self.cost_val.append(self.sess.run(self.cost,feed_dict={self.X:self.x_data,self.Y:self.y_data}))
		#	except:
		#		''''''
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

	#값을 넣엇을 때 라벨이 무엇이 나오나 보여준다.
	def predict(self,x_data):

			self.Y_val.append(self.sess.run(self.hypothesis, feed_dict={self.X: x_data, self.Y: self.y_data}))
			self.X_val.append(x_data)

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

		

	
