import tensorflow as tf

x_data=[1,2,3]
y_data=[1,2,3]

# y_data = w*x_data+b에서 x, b값 랜덤 지정
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 가설함수 H(x)
hypothesis = w * x_data + b

# Cost Function
cost = tf.reduce_mean(tf.squar(hypothesis - y_data))

# Section 3에서 다룰 Cost Minimize
a = tf.Variable(0.1) # Learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# 변수 초기화
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Hypothesis를 회귀에 맞추기
for step in range(2001):
    sess.run(train)
    if step%20 == 0:
        print (step, sess.run(cost), sess.run(w), sess.run(b))