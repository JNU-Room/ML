import tensorflow as ts

x_data = [1.,2.,3.]
y_data = [1.,2.,3.]

W = ts.Variable(ts.random_uniform([1],-1.0,1.0))
b = ts.Variable(ts.random_uniform([1],-1.0,1.0))

hypothesis = W * x_data + b
cost = ts.reduce_mean(ts.square(hypothesis-y_data))

rate = ts.Variable(0.1)
optimizer = ts.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

init = ts.initialize_all_variables()

sess = ts.Session()
sess.run(init)

for step in range(3001):
    sess.run(train)
    if step % 20 == 0:
        print('{:4}{}{}{}'.format(step, sess.run(cost),sess.run(W),sess.run(b)))