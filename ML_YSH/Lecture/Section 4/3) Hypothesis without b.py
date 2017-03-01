import  tensorflow as tf

x_data = [ [1., 1., 1., 1., 1.], # for b
           [1., 0., 3., 0., 5.],
           [0., 2., 0., 4., 0.] ] # 2 * n 행렬
y_data = [1, 2, 3, 4, 5]

# Try to find values for W and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0)) # 1 * 3 행렬 (b를 포함)
# without b # b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Our hypothesis
hypothesis = tf.matmul(W, x_data) # 행렬곱

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
a = tf.Variable(0.1) # Learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# initialize the variables
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line
print ("step, cost, W")

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(cost), sess.run(W))