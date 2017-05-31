# Lab 10 MNIST and Xavier
import tensorflow as tf
from lib.mnist_neural_network import MnistNeuralNetwork
from lib.nntype import NNType


#XAIVER 초기화를 하고자 한다면 다른 건 다 똑같고 다만 가상함수를 재정의하라.
class XXX (MnistNeuralNetwork):
    def set_weight_initializer(self):  # This hole is for you!
        self.xavier()

    def init_network(self):
        self.set_placeholder(784, 10)

        L1 = self.fully_connected_layer(self.X, 784, 256, 'Wa', 'ba')
        L1 = tf.nn.relu(L1)

        L2 = self.fully_connected_layer(L1, 256, 256, 'wb', 'bb')
        L2 = tf.nn.relu(L2)

        L3 = self.fully_connected_layer(L2, 256, 10, 'Wc', 'bc')
        self.set_hypothesis(L3)

        self.set_cost_function(NNType.SOFTMAX_LOGITS)
        self.set_optimizer(NNType.ADAM, 0.001)


gildong = XXX()
gildong.learn_mnist(3, 100)
gildong.evaluate()
gildong.classify_random()

'''
Epoch: 0001 cost = 0.301498963
Epoch: 0002 cost = 0.107252513
Epoch: 0003 cost = 0.064888892
Epoch: 0004 cost = 0.044463030
Epoch: 0005 cost = 0.029951642
Epoch: 0006 cost = 0.020663404
Epoch: 0007 cost = 0.015853033
Epoch: 0008 cost = 0.011764387
Epoch: 0009 cost = 0.008598264
Epoch: 0010 cost = 0.007383116
Epoch: 0011 cost = 0.006839140
Epoch: 0012 cost = 0.004672963
Epoch: 0013 cost = 0.003979437
Epoch: 0014 cost = 0.002714260
Epoch: 0015 cost = 0.004707661
Learning Finished!
Accuracy: 0.9783
'''
