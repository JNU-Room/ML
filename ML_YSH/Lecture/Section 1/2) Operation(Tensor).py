# Tensor의 개념을 잡기 위한 예제
# 2+3값을 출력

import tensorflow as tf
# Start tf session
sess = tf.Session()

# Start constant operation
# The value returned by the constructor represents the output of the const op
a = tf.constant(2)
b = tf.constant(3)

c = a+b

# Print operation(Tensor)
print (c)

# Print out the result of operation
print (sess.run(c))