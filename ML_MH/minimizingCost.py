import tensorflow as ts

X = [1.,2.,3.]
Y = [1.,2.,3.]
m = len(X)
W = ts.placeholder(ts.float32)

hypothesis = ts.mul(W,X)
cost = ts.reduce_sum(ts.pow(hypothesis-Y,2))/m

init = ts.initialize_all_variables()

sess = ts.Session()
sess.run(init)


W_val, cost_val = [], []


for i in range(-30, 51):
    xPos = i*0.1
    yPos = sess.run(cost, feed_dict={W: xPos})

    print('{:3.1f}, {:3.1f}'.format(xPos, yPos))

    W_val.append(xPos)
    cost_val.append(yPos)

sess.close()



import matplotlib.pyplot as plt

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('cost')
plt.xlabel('W')
plt.show()