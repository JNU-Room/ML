import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# 나중에 모델을 업데이트할 수 있게 하기 위해 tf가 지정한 Varialble로 지정 ????
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = w * X + b

# (가설 - 값 )의 제곱의 평균
# TF에서 이렇게한다고 계산이 일어나는 것은 아님, 식만 정해두고 따로 실행시켜야한다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# cost를 Minimize하는 부분
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# 변수 초기화
init = tf.initialize_all_variables()

# 초기화된 변수를 세션에서 제일 먼저 실행시켜 줄
sess = tf.Session()

sess.run(init)

# writer = tf.train.SummaryWriter("/home/voidblueserver/Desktop/tflog", sess.graph)


for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if (step % 20 == 0):
        print('{},{},{},{}'.format(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(w), sess.run(b)))

print('{}'.format(sess.run(hypothesis, feed_dict={X: 5})))
print('{}'.format(sess.run(hypothesis, feed_dict={X: 2.5})))
