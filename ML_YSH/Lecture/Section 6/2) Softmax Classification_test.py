import tensorflow as tf
import numpy as np

# Load Data
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

# tf Graph Input
X = tf.placeholder("float", [None, 3]) # x1, x2 and 1 (for bias)
Y = tf.placeholder("float", [None, 3]) # A, B, C => 3 classes
# Set model weights
W = tf.Variable(tf.zeros([3, 3]))
# Construct model
hypothesis = tf.nn.softmax(tf.matmul(X, W)) # Softmax

# Minimize error using cross entropy
learning_rate = 0.001

# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize all variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    print("step, cost, W")
    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

    print ("[a일 확률, b일 확률, c일 확률]")

    a = sess.run(hypothesis, feed_dict={X: [ [1, 11, 7] ]})
    print ("a, 최댓값 index")
    print (a, sess.run(tf.arg_max(a,1)))

    b = sess.run(hypothesis, feed_dict={X: [ [1, 3, 4] ]})
    print ("b, 최댓값 index")
    print (b, sess.run(tf.arg_max(b ,1)))

    c = sess.run(hypothesis, feed_dict={X: [ [1, 1, 0] ]})
    print ("c, 최댓값 index")
    print (c, sess.run(tf.arg_max(c,1)))

    all = sess.run(hypothesis, feed_dict={X: [ [1, 11, 7], [1, 3, 4], [1, 1, 0] ]})
    print ("all, 최댓값 index")
    print (all, sess.run(tf.arg_max(all,1)))