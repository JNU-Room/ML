import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph
with tf.Session() as sess:
    print("a=2, b=3")
    print ("상수를 활용한 덧셈 : %i" % sess.run(a+b))
    print ("상수를 활용한 곱셈 : %i" % sess.run(a*b))