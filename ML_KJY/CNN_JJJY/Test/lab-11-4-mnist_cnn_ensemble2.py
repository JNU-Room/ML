# Lab 11 MNIST and Deep learning CNN
import tensorflow as tf
from lib.ensemble.ensemble_core import EnsembleCore
from lib.ensemble.mnist_core import MnistCore
from lib.ensemble.cnn_core import CNNCore


class MyCNN (CNNCore):
    def init_network(self):
        self.set_placeholder(784, 10, 28, 28)
        self.DO = tf.placeholder(tf.float32)

        L1 = self.convolution_layer(self.X_2d, 3, 3, 1, 32, 1, 1)
        L1 = self.relu(L1)

        L1_maxpool = self.max_pool(L1, 2, 2, 2, 2)
        L1_maxpool = self.dropout(L1_maxpool)

        L2 = self.convolution_layer(L1_maxpool, 3, 3, 32, 64, 1, 1)
        L2 = self.relu(L2)

        L2_maxpool = self.max_pool(L2, 2, 2, 2, 2)
        L2_maxpool = self.dropout(L2_maxpool)

        L3 = self.convolution_layer(L2_maxpool, 3, 3, 64, 128, 1, 1)
        L3 = self.relu(L3)

        L3_maxpool = self.max_pool(L3, 2, 2, 2, 2)
        L3_maxpool = self.dropout(L3_maxpool)

        # L4 FC 4x4x128 inputs -> 625 outputs
        reshaped = tf.reshape(L3_maxpool, [-1, 128 * 4 * 4])
        L4 = self.fully_connected_layer(reshaped, 128 * 4 * 4, 625, 'W4')
        L4 = self.relu(L4)
        L4 = self.dropout(L4)

        self.logit = self.fully_connected_layer(L4, 625, 10, 'W5')

        self.set_hypothesis(self.logit)
        self.set_cost_function()
        self.set_optimizer(0.001)


class MyEnsemble (EnsembleCore):
    mnist = MnistCore()

    def load_db(self):
        self.mnist.load_mnist()

    def set_networks(self, sess, num_of_network):
        self.create_networks(sess, MyCNN, 'network_name', 7)

    def get_number_of_segment(self, seg_size):
        return self.mnist.get_number_of_segment(seg_size)

    def get_next_segment(self, seg_size):
        return self.mnist.get_next_segment(seg_size)

    def get_test_data(self):
        return self.mnist.get_test_x_data(), self.mnist.get_test_y_data()


gildong = MyEnsemble()
gildong.learn_ensemble(7, 15, 100)
gildong.evaluate_all_models()


'''
0 Accuracy: 0.9933
1 Accuracy: 0.9946
2 Accuracy: 0.9934
3 Accuracy: 0.9935
4 Accuracy: 0.9935
5 Accuracy: 0.9949
6 Accuracy: 0.9941

Ensemble accuracy: 0.9952
'''
