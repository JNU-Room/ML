import sys
import random
import numpy as np

#학습하는 동안 진행되는 것을 표시하기 위하여 점을 한줄로 찍도록 하는 코드. print('.')는 새로운 줄에 찍어버림.
def print_dot():
    sys.stdout.write('.')
    sys.stdout.flush()

def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

def get_random_int(max):
    return random.randint(0, max - 1)

def printf():
    msg = "{} {:.6f} {} {}".format(1, 0.693147, 9.64292212e-07, 9.65349955e-07)
    print(msg)

def get_numpy_data():
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
    return x_data, y_data

# x_col = len(x_data[0])
# y_col = len(y_data[0])
# print(x_col, y_col) # 3, 1



