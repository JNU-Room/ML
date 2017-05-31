# Lab 10 MNIST and Deep learning
import tensorflow as tf
from lib.mnist_neural_network import MnistNeuralNetwork
from lib.nntype import NNType


'''
in:image (784) -> out:0~9 (10)
5 layers
Xavier initialization
RELU activation function
Hypothesis: Softmax, cf) One-hot encoding (argmax)
Cost function: Cross-Entropy, D(H,Y)
Optimizer: ADAM
'''


class XXX (MnistNeuralNetwork):
    def set_weight_initializer(self):
        self.xavier()

    def init_network(self):
        self.set_placeholder(784, 10)

        L1 = self.fully_connected_layer(self.X, 784, 512, 'weight_a', 'bias_a')
        L1 = tf.nn.relu(L1)

        L2 = self.fully_connected_layer(L1, 512, 512, 'weight_b', 'bias_b')
        L2 = tf.nn.relu(L2)

        L3 = self.fully_connected_layer(L2, 512, 512, 'weight_c', 'bias_c')
        L3 = tf.nn.relu(L3)

        L4 = self.fully_connected_layer(L3, 512, 512, 'weight_d', 'bias_d')
        L4 = tf.nn.relu(L4)

        hypo = self.fully_connected_layer(L4, 512, 10, 'weight_e', 'bias_e')

        self.set_hypothesis(hypo)
        self.set_cost_function(NNType.SOFTMAX_LOGITS)
        self.set_optimizer(NNType.ADAM, 0.001)



gildong = XXX()
gildong.learn_mnist(1, 100)
gildong.evaluate()
#gildong.classify_random()
gildong.show_error()

