import tensorflow as tf

#tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

#Define some operations
add = tf.add(a,b)
mul = tf.mul(a,b)

#Launch the default graph
with tf.Session() as sess:
    #Run every operation with variable input
    print ("변수들의 합 : %i" % sess.run(add, feed_dict={a:2,b:3}))
    print ("변수들의 합 : %i" % sess.run(mul, feed_dict={a:2,b:3}))
