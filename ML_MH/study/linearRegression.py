import tensorflow as ts

x_data = [1.,2.,3.,4.]
y_data = [1.,2.,3.,4.]
# test = ts.Variable(ts.random)
# test = ts.Variable(ts.random) git modifiy
W = ts.Variable(ts.random_uniform([1],-100,100))
b = ts.Variable(ts.random_uniform([1],-100,100))
X = ts.placeholder(ts.float32)
Y = ts.placeholder(ts.float32)
hypothesis = W * X + b
cost = ts.reduce_mean(ts.square(hypothesis-Y))

rate = ts.Variable(0.1)
optimizer = ts.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

init = ts.initialize_all_variables()

sess = ts.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

        #print(sess.run(hypothesis,feed_dict={X:5}))
        #print(sess.run(hypothesis,feed_dict={X:2.5}))
        print(sess.run(hypothesis,feed_dict={X:[2.5,5]}))