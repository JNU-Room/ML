# TensorFlow를 활용하여 "Hello world" 출력하기

import tensorflow as tf

hello = tf.constant('Hello world')
sess = tf.Session();

print (sess.run(hello)) # Hello world 출력
print (hello) # Tensor의 정보 출력