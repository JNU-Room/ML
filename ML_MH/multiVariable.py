import tensorflow as ts

x1_data = [1,0,3,0,5]
x2_data = [0,2,0,4,0]
y_data = [1,2,3,4,5]

W1 = ts.Variable(ts.random_uniform([1],-1.0,1.0))
W2 = ts.Variable(ts.random_uniform([1],-1.0,1.0))

b = ts.Variable(ts.random_uniform([1],-1.0,1.0))

hypothesis = W1*x1_data +W2*x2_data+b
cost = ts.reduce_mean(ts.square(hypothesis-y_data))
rate = ts.Variable(0.1)
optimizer = ts.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

init = ts.initialize_all_variables()

sess = ts.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)

    if step%20 == 0:
        print(step,sess.run(cost),sess.run(W1),sess.run(W2),sess.run(b))

sess.close()