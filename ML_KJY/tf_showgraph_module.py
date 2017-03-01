import tensorflow as tf
import matplotlib.pyplot as plt

class LenearRegession:
	W_val = []
	cost_val = []
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


	def __init__(self,x_data,y_data):
		i=0
		a = x_data
		self.sess = tf.Session()
		self.X = tf.placeholder(tf.float32)
		self.Y = tf.placeholder(tf.float32)
		m = len(x_data)
		self.x_data = x_data
		self.y_data = y_data
		while(1):
			try:
				a = a[0]
				i += 1
				print(i)

			except:
				break
		if i != 1:
			self.W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
		else:
			self.W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
		self.b = tf.Variable(tf.random_uniform([1],-1.0,1.0))




		try:
			self.hypothesis = tf.matmul(self.X,self.W) + self.b
		except:
			self.hypothesis = self.X*self.W + self.b

		self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y))



		init = tf.initialize_all_variables()
		self.sess.run(init)
		print(self.sess.run(self.hypothesis,feed_dict={self.X:self.x_data,self.Y:self.y_data}))
	def training(self,learning_rate=0.04, step = 2001):
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.train = self.optimizer.minimize(self.cost)

		for step in range(step):
			self.sess.run(self.train,feed_dict={self.X:self.x_data,self.Y:self.y_data})
			self.W_val.append(self.sess.run(self.W,feed_dict={self.X:self.x_data}))
			self.cost_val.append(self.sess.run(self.cost,feed_dict={self.X:self.x_data,self.Y:self.y_data}))

	def show_cost_graph(self):
		plt.plot(self.W_val, self.cost_val, 'ro')
		plt.ylabel('cost')
		plt.xlabel('weight')
		plt.show()


	def predict(self,x_data):
			print(self.sess.run(self.hypothesis,feed_dict={self.X:x_data}))

		

	
